#!/usr/bin/env python
"""Benchmark: torch_dtype comparison (float32 vs float16 vs bfloat16).

Each dtype runs in its own subprocess for a clean state.

Note: bfloat16 only supported on gfx1101+ (RDNA3). float16 may cause NaN
on some AMD cards — the script checks audio validity.

Usage:
  conda run -n qwen3-tts python bench_dtype.py
"""

import os
import subprocess
import sys
import time

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_dtype_worker.py")

CONFIGS = [
    ("float32 (default)", "float32"),
    ("float16", "float16"),
    ("bfloat16", "bfloat16"),
]


def main():
    python = sys.executable

    for i, (label, dtype) in enumerate(CONFIGS):
        print()
        print("=" * 70)
        print(f"  CONFIG {i+1}/{len(CONFIGS)}: dtype={label}")
        print("=" * 70)

        env = os.environ.copy()
        env["BENCH_DTYPE"] = dtype

        t0 = time.time()
        proc = subprocess.run(
            [python, WORKER],
            env=env,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        wall = time.time() - t0

        print(proc.stdout)

        if proc.stderr:
            warn_lines = [l for l in proc.stderr.splitlines()
                          if "warning" in l.lower() or "Warning" in l]
            err_lines = [l for l in proc.stderr.splitlines()
                         if "error" in l.lower() or "Error" in l or "nan" in l.lower() or "NaN" in l]
            if err_lines:
                print(f"  !! ERRORS / NaN ({len(err_lines)}):")
                for e in err_lines[:10]:
                    print(f"     {e[:200]}")
            if warn_lines:
                print(f"  Warnings ({len(warn_lines)}):")
                for w in warn_lines[:10]:
                    print(f"     {w[:160]}")
        else:
            print("  stderr clean")

        print(f"  Wall clock: {wall:.0f}s")

    print()
    print("=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print("  Key: bench_RTF (lower=better), vram_peak, audio_ok")


if __name__ == "__main__":
    main()
