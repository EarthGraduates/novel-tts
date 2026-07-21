#!/usr/bin/env python
"""Benchmark MIOPEN_FIND_MODE=2 for Qwen3-TTS on ROCm.

Usage:
  # baseline
  conda run -n qwen3-tts python bench_miopen.py

  # with optimization
  MIOPEN_FIND_MODE=2 TORCH_BLAS_PREFER_HIPBLASLT=0 \
    conda run -n qwen3-tts python bench_miopen.py
"""
import os
import time
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import torch

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
OUTPUT = "/home/phoenix/ClaudeProjects/TTS"

TEST_CASES = [
    ("short", "你好，我是你的语音助手。", "Vivian", None),
    ("medium", "夜幕降临，城市的灯光陆续亮起。张建国站在办公室的窗前，望着远处的车水马龙。", "Uncle_Fu", "narration"),
    ("long",
     "张建国是一名普通的中学历史教师，在这座北方小城生活了整整四十年。"
     "他的学生遍布各行各业，有的成了医生，有的做了工程师，还有几个在市里当了干部。"
     "每到春节，那些已经毕业多年的学生总会三三两两地来给他拜年。",
     "Uncle_Fu", "narration"),
]


def run_bench():
    env_flag = os.environ.get("MIOPEN_FIND_MODE", "1 (default)")
    print(f"MIOPEN_FIND_MODE = {env_flag}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"ROCm: {torch.version.hip}")
    print("=" * 60)

    # Load model
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0")
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    results = []
    # Two passes: second pass is after warmup
    for round_num, label in enumerate(["Warmup", "Bench"], 1):
        print(f"--- {label} (round {round_num}) ---")
        for name, text, speaker, instruct in TEST_CASES:
            t1 = time.time()
            audios, sr = model.generate_custom_voice(
                text=text, speaker=speaker, language="chinese", instruct=instruct,
            )
            elapsed = time.time() - t1
            chars = len(text)
            rtf = elapsed / (len(audios[0]) / sr)  # RTF: >1 slower than realtime, <1 faster
            print(f"  {name:6s} | {chars:3d} chars | {elapsed:6.1f}s | {chars / elapsed:.2f} ch/s | RTF {rtf:.2f}")

            if round_num == 2:
                results.append((name, chars, elapsed, rtf))

            fname = f"{OUTPUT}/bench_{name}_{'opt' if env_flag == '2' else 'base'}_r{round_num}.wav"
            sf.write(fname, audios[0], sr)

    print()
    print("=" * 60)
    print("Summary (after warmup):")
    for name, chars, elapsed, rtf in results:
        print(f"  {name:6s}: {elapsed:.1f}s, {chars / elapsed:.2f} ch/s, RTF {rtf:.2f}")


if __name__ == "__main__":
    run_bench()
