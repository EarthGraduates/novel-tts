#!/usr/bin/env python
"""Benchmark: Flash Attention / SDPA configurations for gfx1101 (RDNA3).

Tests multiple env-var combos by calling bench_sdpa_worker.py in subprocesses
so env vars take effect BEFORE torch imports.

Usage:
  conda run -n qwen3-tts python bench_sdpa_experimental.py
"""

import os
import subprocess
import sys
import time

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_sdpa_worker.py")

CONFIGS = [
    ("baseline", {}),
    (
        "SDPA-experimental",
        {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"},
    ),
    (
        "SDPA-experimental + Autotune",
        {
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
            "FLASH_ATTENTION_TRITON_AMD_AUTOTUNE": "TRUE",
        },
    ),
]


def main():
    python = sys.executable

    for i, (label, env_overrides) in enumerate(CONFIGS):
        print()
        print("=" * 70)
        print(f"  CONFIG {i+1}/{len(CONFIGS)}: {label}")
        print("=" * 70)

        env = os.environ.copy()
        env.update(env_overrides)

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

        # ── Warnings ──
        if proc.stderr:
            warn_lines = [l for l in proc.stderr.splitlines()
                          if "warning" in l.lower() or "Warning" in l]
            if warn_lines:
                print(f"  Warnings ({len(warn_lines)}):")
                for w in warn_lines[:15]:
                    print(f"     {w[:160]}")
                if len(warn_lines) > 15:
                    print(f"     ... and {len(warn_lines) - 15} more")
            else:
                first = proc.stderr.strip().splitlines()[:3]
                if first:
                    print(f"  stderr ({len(proc.stderr.splitlines())} lines):")
                    for line in first:
                        print(f"     {line[:160]}")
        else:
            print("  stderr clean")

        print(f"  Wall clock: {wall:.0f}s")

    print()
    print("=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print("  Compare the SUMMARY blocks above across configs.")
    print("  Key metrics: bench_RTF, bench_total, vram_peak")


if __name__ == "__main__":
    main()
