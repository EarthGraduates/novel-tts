#!/usr/bin/env python
"""Worker script — called by bench_sdpa_experimental.py with env vars pre-set.

Do NOT run directly; the parent script sets env vars before spawning this.
"""

import os, time, sys
import numpy as np

# ── Fixed defaults (always set before torch import) ──
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.1")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import torch
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

TEST_CASES = [
    ("short",  "你好，我是你的语音助手。", "Vivian"),
    ("medium", "夜幕降临，城市的灯光陆续亮起。张建国站在办公室的窗前，望着远处的车水马龙，心中涌起一股说不清的感慨。", "Uncle_Fu"),
    ("long",
     "张建国是一名普通的中学历史教师，在这座北方小城生活了整整四十年。"
     "他的学生遍布各行各业，有的成了医生，有的做了工程师，还有几个在市里当了干部。"
     "每到春节，那些已经毕业多年的学生总会三三两两地来给他拜年。"
     "这一年的秋天，学校来了一位年轻的实习老师，姓李，刚从省城的师范大学毕业。",
     "Uncle_Fu"),
]

# ── All env vars relevant to ROCm / Flash Attention ──
RELEVANT_ENV = [
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
    "FLASH_ATTENTION_TRITON_AMD_AUTOTUNE",
    "HSA_OVERRIDE_GFX_VERSION",
    "HIP_VISIBLE_DEVICES",
    "MIOPEN_FIND_MODE",
    "TORCH_BLAS_PREFER_HIPBLASLT",
]


def print_params():
    """Print all relevant parameters in a compact table."""
    print("-" * 62)
    print(f"  {'Parameter':<44s} {'Value':>15s}")
    print("-" * 62)
    for k in RELEVANT_ENV:
        v = os.environ.get(k, "(not set)")
        print(f"  {k:<44s} {v:>15s}")
    print(f"  {'GPU':<44s} {torch.cuda.get_device_name(0):>15s}")
    print(f"  {'PyTorch':<44s} {torch.__version__:>15s}")
    print(f"  {'ROCm':<44s} {torch.version.hip:>15s}")
    print(f"  {'flash_sdp_enabled':<44s} {str(torch.backends.cuda.flash_sdp_enabled()):>15s}")
    print(f"  {'mem_efficient_sdp_enabled':<44s} {str(torch.backends.cuda.mem_efficient_sdp_enabled()):>15s}")
    print(f"  {'math_sdp_enabled':<44s} {str(torch.backends.cuda.math_sdp_enabled()):>15s}")
    print("-" * 62)
    print()


def run_bench():
    print_params()

    # Load model
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0")
    load_t = time.time() - t0
    vram_after_load = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded in {load_t:.1f}s")
    print(f"VRAM after load: {vram_after_load:.2f} GB")
    print()

    # Warmup (before compile)
    print("Pre-compile warmup...", end=" ", flush=True)
    t0 = time.time()
    model.generate_custom_voice(text="你好。", speaker="Vivian", language="chinese")
    warmup_pre = time.time() - t0
    print(f"{warmup_pre:.1f}s")

    # Compile (same as production)
    print("torch.compile(reduce-overhead)...", end=" ", flush=True)
    t0 = time.time()
    model.model = torch.compile(model.model, mode="reduce-overhead")
    compile_t = time.time() - t0
    print(f"{compile_t:.1f}s")

    # Compile warmup
    print("Post-compile warmup...", end=" ", flush=True)
    t0 = time.time()
    model.generate_custom_voice(text="你好。", speaker="Vivian", language="chinese")
    warmup_post = time.time() - t0
    print(f"{warmup_post:.1f}s")

    # ── Benchmark ──
    print()
    print(f"  {'case':<8s} {'chars':>5s} {'time':>7s} {'ch/s':>7s} {'RTF':>6s} {'audio':>7s}")
    print(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*7} {'-'*6} {'-'*7}")
    results = []
    for name, text, speaker in TEST_CASES:
        torch.cuda.synchronize()
        t0 = time.time()
        audios, sr = model.generate_custom_voice(text=text, speaker=speaker, language="chinese")
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        chars = len(text)
        audio_dur = len(audios[0]) / sr
        rtf = elapsed / audio_dur
        results.append((name, chars, elapsed, rtf, audio_dur))
        print(f"  {name:<8s} {chars:>5d} {elapsed:>6.1f}s {chars/elapsed:>6.1f} {rtf:>5.2f} {audio_dur:>6.1f}s")

    total_audio = sum(r[4] for r in results)
    total_time = sum(r[2] for r in results)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*7} {'-'*6} {'-'*7}")
    print(f"  {'TOTAL':<8s} {'':>5s} {total_time:>6.1f}s {'':>7s} {total_time/total_audio:>5.2f} {total_audio:>6.1f}s")

    # ── Summary block ──
    print()
    print("+--------------------------------------------------------------+")
    print("|  SUMMARY                                                     |")
    print("+--------------------------------------------------------------+")
    print(f"|  load:          {load_t:>6.1f}s                                       |")
    print(f"|  warmup_pre:    {warmup_pre:>6.1f}s                                       |")
    print(f"|  compile:       {compile_t:>6.1f}s                                       |")
    print(f"|  warmup_post:   {warmup_post:>6.1f}s                                       |")
    print(f"|  bench_total:   {total_time:>6.1f}s                                       |")
    print(f"|  bench_RTF:     {total_time/total_audio:>6.2f}                                        |")
    print(f"|  vram_load:     {vram_after_load:>5.2f} GB                                      |")
    print(f"|  vram_peak:     {peak_vram:>5.2f} GB                                      |")
    print("+--------------------------------------------------------------+")


if __name__ == "__main__":
    run_bench()
