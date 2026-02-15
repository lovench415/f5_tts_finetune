#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_tts.py
===============

Единый CLI‑инструмент для:
  • подготовки датасета формата csv+wavs → HuggingFace Arrow
  • тонкой настройки (fine‑tune) модели F5‑TTS

▪ Запустите без аргументов — скрипт задаст вопросы пошагово.
▪ Нажмите Enter, чтобы принять значение по умолчанию (в [квадратных скобках]).
▪ Директории `data_prepared/` и `ckpts/` создаются рядом с этим скриптом.

Примеры запуска
---------------
# Полностью интерактивно (prepare + train)
$ python finetune_tts.py

# Только подготовка датасета:
$ python finetune_tts.py prepare \
      --inp_dir /path/to/csv_wavs \
      --out_dir /path/to/data_prepared \
      --vocab_path /path/to/vocab.txt

# Полный цикл без вопросов:
$ python finetune_tts.py all \
      --inp_dir /path/to/csv_wavs \
      --vocab_path /path/to/vocab.txt \
      --ckpt /path/to/base_model.pt \
      --epochs 1 \
      --lr 1e-5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from accelerate import Accelerator
from datasets import Dataset as HFDataset_
from datasets import load_from_disk
from datasets.arrow_writer import ArrowWriter
from f5_tts.model import CFM, DiT, Trainer
from f5_tts.model.dataset import CustomDataset
from f5_tts.model.utils import (
    convert_char_to_pinyin,
    get_tokenizer,
    list_str_to_idx,
)
from tqdm import tqdm

# ────────────────────────────────────────────────
# 1. Базовые константы и дефолт‑директории
# ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PREP_DIR = BASE_DIR / "data_prepared"
DEFAULT_CKPT_DIR = BASE_DIR / "ckpts"

# ────────────────────────────────────────────────
# 2. Настройка окружения (NCCL, PyTorch, torch / accelerate)
# ────────────────────────────────────────────────
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1200"

# ────────────────────────────────────────────────
# 3. Утилиты подготовки датасета (csv+wavs → arrow)
# ────────────────────────────────────────────────
BATCH_SIZE = 100
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)
THREAD_NAME_PREFIX = "AudioProcessor"
CHUNK_SIZE = 100
executor: concurrent.futures.ThreadPoolExecutor | None = None


@contextmanager
def graceful_exit():
    """Корректное завершение по Ctrl‑C / SIGTERM."""
    def _handler(signum, frame):
        print("\n⛔  Interrupt received, cleaning up…")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    try:
        yield
    finally:
        if executor is not None:
            executor.shutdown(wait=False)


def is_csv_wavs_format(dataset_dir: os.PathLike) -> bool:
    p = Path(dataset_dir)
    return (p / "metadata.csv").is_file() and (p / "wavs").is_dir()


def get_audio_duration(path: str, timeout: int = 5) -> float:
    """Сначала ffprobe (быстро), fallback — torchaudio."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=timeout,
        )
        return float(res.stdout.strip())
    except Exception:
        audio, sr = torchaudio.load(path)
        return audio.shape[1] / sr


def read_audio_text_pairs(csv_file: os.PathLike) -> List[Tuple[str, str]]:
    root = Path(csv_file).parent
    pairs: List[Tuple[str, str]] = []
    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f, delimiter="|")
        next(rdr, None)  # пропускаем заголовок
        for row in rdr:
            if len(row) >= 2:
                fil_nm = row[0].strip()+".mp3"
                pairs.append((str(root / "wavs" / fil_nm), row[1].strip()))
    return pairs


def batch_convert_texts(
    texts: List[str], polyphone: bool = True, batch_size: int = BATCH_SIZE
) -> List[str]:
    out: List[str] = []
    for i in range(0, len(texts), batch_size):
        out.extend(
            convert_char_to_pinyin(texts[i : i + batch_size], polyphone=polyphone)
        )
    return out


def process_audio_file(audio_path: str, text: str, polyphone: bool = True):
    """Возвращает (audio_path, converted_text, duration) либо None при ошибке."""
    if not Path(audio_path).exists():
        return None
    try:
        dur = get_audio_duration(audio_path)
        if dur <= 0:
            raise ValueError("duration <= 0")
        return audio_path, text, dur
    except Exception:
        return None


def prepare_csv_wavs_dir(inp_dir, num_workers=None):
    global executor
    if not is_csv_wavs_format(inp_dir):
        raise ValueError(f"{inp_dir} is not in csv_wavs format")

    pairs = read_audio_text_pairs(Path(inp_dir) / "metadata.csv")
    total = len(pairs)
    workers = num_workers if num_workers else min(MAX_WORKERS, total)
    print(f"🛠  Processing {total} files on {workers} threads")

    results = []
    with graceful_exit(), concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix=THREAD_NAME_PREFIX
    ) as executor:
        futs = [executor.submit(process_audio_file, p, t) for p, t in pairs]
        for fut in tqdm(
            concurrent.futures.as_completed(futs), total=len(futs), desc="audio"
        ):
            r = fut.result()
            if r:
                results.append(r)

    if not results:
        raise RuntimeError("No valid audio processed!")

    raw_texts = [t for _, t, _ in results]
    converted = batch_convert_texts(raw_texts)

    entries, durations, vocab = [], [], set()
    for (audio, _t, dur), conv in zip(results, converted):
        entries.append({"audio_path": audio, "text": conv, "duration": dur})
        durations.append(dur)
        vocab.update(conv)

    return entries, durations, vocab


def save_prepared_dataset(
    out_dir,
    entries,
    durations,
    vocab_set,
    is_finetune: bool,
    vocab_path: str | None,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving dataset → {out}")

    with ArrowWriter(path=str(out / "raw.arrow"), writer_batch_size=100) as w:
        for e in tqdm(entries, desc="arrow"):
            w.write(e)

    (out / "duration.json").write_text(
        json.dumps({"duration": durations}, ensure_ascii=False)
    )

    vocab_out = out / "vocab.txt"
    if is_finetune:
        if not vocab_path or not Path(vocab_path).is_file():
            raise FileNotFoundError("Pretrained vocab.txt required for finetune")
        shutil.copy2(vocab_path, vocab_out)
    else:
        with open(vocab_out, "w", encoding="utf-8") as f:
            for v in sorted(vocab_set):
                f.write(v + "\n")

    print(
        f"📊 samples: {len(entries)} | vocab: {len(vocab_set)} | hours: {sum(durations)/3600:.2f}"
    )


def prepare_and_save_set(
    inp_dir,
    out_dir,
    vocab_path,
    is_finetune: bool = True,
    num_workers: int | None = None,
):
    entries, durs, vocab = prepare_csv_wavs_dir(inp_dir, num_workers)
    save_prepared_dataset(out_dir, entries, durs, vocab, is_finetune, vocab_path)


# ────────────────────────────────────────────────
# 4. Fine‑tune F5‑TTS
# ────────────────────────────────────────────────
def run_finetune(
    prepared_dir,
    output_dir,
    vocab_path,
    ckpt_path,
    epochs: int,
    lr: float,
    batch_size_frames: int,
):
    accelerator = Accelerator(mixed_precision="fp16")
    print(f"⚡  device: {accelerator.device}")

    # Tokenizer
    vocab_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    tokenizer_fn = lambda txts: list_str_to_idx(txts, vocab_map)  # noqa: E731
    if accelerator.is_main_process:
        print(f"Vocab size: {vocab_size}")

    # Model
    mel_args = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )
    model = CFM(
        transformer=DiT(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            text_num_embeds=vocab_size,
        ),
        mel_spec_kwargs=mel_args,
        vocab_char_map=vocab_map,
    )

    # Load checkpoint
    print("🔄 Loading base checkpoint…")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state), strict=False)

    # Dataset
    prepared_dir = Path(prepared_dir)
    try:
        ds_raw = load_from_disk(str(prepared_dir / "raw"))
    except Exception:
        ds_raw = HFDataset_.from_file(str(prepared_dir / "raw.arrow"))
    durations = json.loads((prepared_dir / "duration.json").read_text())["duration"]

    dataset = CustomDataset(
        ds_raw, durations=durations, preprocessed_mel=False, **mel_args
    )

    # Trainer
    trainer = Trainer(
        model=model,
        epochs=epochs,
        learning_rate=lr,
        num_warmup_updates=26666,
        save_per_updates=2000,
        keep_last_n_checkpoints=6,
        checkpoint_path=str(output_dir),
        batch_size_per_gpu=batch_size_frames,
        batch_size_type="frame",
        max_samples=64,
        grad_accumulation_steps=1,
        max_grad_norm=1,
        logger="tensorboard" if accelerator.is_main_process else None,
        wandb_project=prepared_dir.name if accelerator.is_main_process else None,
        wandb_run_name="finetune",
        last_per_updates=10000,
        mixed_precision="bf16",
    )

    trainer.model = accelerator.prepare(trainer.model)

    if accelerator.is_main_process:
        print("🚀 Starting fine‑tune…")
    trainer.train(dataset)
    if accelerator.is_main_process:
        print("✅ Fine‑tune complete.")


# ────────────────────────────────────────────────
# 5. CLI
# ────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description="Prepare csv+wavs and fine‑tune F5‑TTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="subcmd")

    # prepare
    sp = sub.add_parser("prepare", help="only prepare dataset")
    sp.add_argument("--inp_dir", help="csv+wavs dataset directory")
    sp.add_argument("--out_dir", default=str(DEFAULT_PREP_DIR))
    sp.add_argument(
        "--pretrain",
        action="store_true",
        help="set if this is NOT a finetune but fresh pre‑training",
    )
    sp.add_argument("--vocab_path", help="pretrained vocab.txt for finetune")
    sp.add_argument("--workers", type=int, help="threads for audio processing")

    # train
    st = sub.add_parser("train", help="only fine‑tune")
    st.add_argument("--prepared_dir", default=str(DEFAULT_PREP_DIR))
    st.add_argument("--vocab_path")
    st.add_argument("--ckpt", help="base checkpoint .pt")
    st.add_argument("--output_ckpts", default=str(DEFAULT_CKPT_DIR))
    st.add_argument("--epochs", type=int, default=100)
    st.add_argument("--lr", type=float, default=1e-5)
    st.add_argument("--batch_size_frames", type=int, default=4000)

    # all
    sa = sub.add_parser("all", help="prepare + fine‑tune")
    for a in sp._actions + st._actions:
        if a.dest not in {x.dest for x in sa._actions}:
            sa._add_action(a)

    return p


def interactive_prompt(args: argparse.Namespace):
    """Запрашиваем недостающие аргументы."""
    print(
        "\n📝  Скрипт выполнит подготовку и/или fine‑tune.\n"
        "⏎  — принять значение по умолчанию (в [квадратных скобках]).\n"
    )

    def ask(attr: str, prompt: str, default: str | None = None):
        val = getattr(args, attr, None)
        if not val:
            ans = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
            setattr(args, attr, ans or default)

    if args.subcmd in ("prepare", "all"):
        ask("inp_dir", "🗂  Path to csv+wavs dataset")
        ask("out_dir", "📁 Output dir for prepared dataset", str(DEFAULT_PREP_DIR))
        if not getattr(args, "pretrain", False):
            ask("vocab_path", "📃 Path to pretrained vocab.txt")

    if args.subcmd in ("train", "all"):
        ask(
            "prepared_dir",
            "🗂  Path to prepared dataset",
            str(getattr(args, "out_dir", DEFAULT_PREP_DIR)),
        )
        ask("ckpt", "🔑 Pretrained checkpoint .pt")
        ask(
            "output_ckpts",
            "📁 Output dir for fine‑tune ckpts",
            str(DEFAULT_CKPT_DIR),
        )
        ask(
            "vocab_path",
            "📃 vocab.txt (tokenizer)",
            str(Path(args.prepared_dir) / "vocab.txt")
            if args.prepared_dir
            else None,
        )
        ask("epochs", "🔄 Epochs", "100")
        ask("lr", "💡 Learning rate", "1e-5")
        ask("batch_size_frames", "📦 Batch (frames)", "4000")


# ────────────────────────────────────────────────
# 6. main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = build_parser()
    ns = parser.parse_args()

    # По умолчанию выполним всё (prepare + train)
    if ns.subcmd is None:
        ns.subcmd = "all"

    # Гарантируем наличие всех атрибутов
    for k in [
        "inp_dir",
        "out_dir",
        "vocab_path",
        "workers",
        "pretrain",
        "prepared_dir",
        "ckpt",
        "output_ckpts",
        "epochs",
        "lr",
        "batch_size_frames",
    ]:
        if not hasattr(ns, k):
            setattr(ns, k, None)

    interactive_prompt(ns)

    # SH‑prepare
    if ns.subcmd in ("prepare", "all"):
        prepare_and_save_set(
            inp_dir=ns.inp_dir,
            out_dir=ns.out_dir,
            vocab_path=ns.vocab_path,
            is_finetune=not ns.pretrain,
            num_workers=ns.workers,
        )

    # SH‑train
    if ns.subcmd in ("train", "all"):
        run_finetune(
            prepared_dir=ns.prepared_dir or ns.out_dir,
            output_dir=ns.output_ckpts,
            vocab_path=ns.vocab_path
            or str(Path(ns.prepared_dir or ns.out_dir) / "vocab.txt"),
            ckpt_path=ns.ckpt,
            epochs=int(ns.epochs),
            lr=float(ns.lr),
            batch_size_frames=int(ns.batch_size_frames),
        )
