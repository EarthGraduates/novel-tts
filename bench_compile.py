#!/usr/bin/env python
"""Benchmark torch.compile modes for Qwen3-TTS (MIOPEN_FIND_MODE=2 already set).

Usage:
  MIOPEN_FIND_MODE=2 TORCH_BLAS_PREFER_HIPBLASLT=0 \
    conda run -n qwen3-tts python bench_compile.py
"""
import os
import time
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import torch

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
OUTPUT = "/home/phoenix/ClaudeProjects/TTS"

# Short text for quick A/B, but long enough to measure
TEST_TEXT = "张建国站在办公室的窗前，望着远处的车水马龙，心中涌起一股说不清的感慨。"
SPEAKER = "Uncle_Fu"
INSTRUCT = "narration"


def load_model(compile_mode=None):
    """Load model, optionally apply torch.compile."""
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0")

    if compile_mode:
        print(f"  Applying torch.compile(mode={compile_mode!r})...")
        model.model = torch.compile(model.model, mode=compile_mode)
        # Warmup compile: first run triggers JIT compilation
        print("  Warming up (compilation run)...")
        t0 = time.time()
        model.generate_custom_voice(
            text="你好。", speaker="Vivian", language="chinese",
        )
        print(f"  Warmup done in {time.time() - t0:.1f}s")

    return model


def run(name, compile_mode, rounds=3):
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")

    model = load_model(compile_mode)

    timings = []
    for r in range(rounds):
        t0 = time.time()
        audios, sr = model.generate_custom_voice(
            text=TEST_TEXT, speaker=SPEAKER, language="chinese", instruct=INSTRUCT,
        )
        elapsed = time.time() - t0
        timings.append(elapsed)
        chars = len(TEST_TEXT)

        fname = f"{OUTPUT}/bench_compile_{name.replace(' ', '_')}_r{r+1}.wav"
        sf.write(fname, audios[0], sr)
        print(f"  Round {r+1}: {chars} chars, {elapsed:.1f}s, {chars/elapsed:.1f} ch/s")

    avg = sum(timings[1:]) / (rounds - 1)  # skip first (includes warmup)
    print(f"  → Average (excl warmup): {avg:.1f}s, {chars/avg:.1f} ch/s")
    return timings, avg


def main():
    print(f"MIOPEN_FIND_MODE = {os.environ.get('MIOPEN_FIND_MODE', 'not set')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Test text ({len(TEST_TEXT)} chars): {TEST_TEXT[:40]}...")

    # ---- Eager (no compile) ----
    eager_times, eager_avg = run("eager", compile_mode=None)

    # ---- torch.compile reduce-overhead ----
    ro_times, ro_avg = run("reduce-overhead", compile_mode="reduce-overhead")

    # ---- torch.compile max-autotune ----
    ma_times, ma_avg = run("max-autotune", compile_mode="max-autotune")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  {'Config':<20s} {'R1':>8s} {'R2':>8s} {'R3':>8s} {'Avg(excl R1)':>14s}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*14}")

    baseline = eager_avg
    for label, times, avg in [
        ("eager (no compile)", eager_times, eager_avg),
        ("reduce-overhead", ro_times, ro_avg),
        ("max-autotune", ma_times, ma_avg),
    ]:
        speedup = baseline / avg if avg > 0 else 0
        print(f"  {label:<20s} {times[0]:7.1f}s {times[1]:7.1f}s {times[2]:7.1f}s {avg:7.1f}s ({speedup:.2f}x)")

    print(f"\nAll audio saved to: {OUTPUT}/bench_compile_*.wav")


if __name__ == "__main__":
    main()
