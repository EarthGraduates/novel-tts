"""Qwen3-TTS model loading and inference utilities.

Uses Qwen3TTSModel with built-in speakers and instruction-based voice control.
No reference audio needed.

Speakers: Vivian, Serena, Ono_Anna, Sohee, Uncle_Fu, Ryan, Aiden, Eric, Dylan
"""

import os
import sys
import warnings
from contextlib import contextmanager

import numpy as np

# ── Verbosity control ──────────────────────────────────────────────────
VERBOSE = os.environ.get("NOVEL_TTS_VERBOSE", "").lower() in ("1", "true", "yes")


@contextmanager
def _quiet():
    """Suppress stdout/stderr within this context, unless VERBOSE is set."""
    if VERBOSE:
        yield
        return
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ── Suppress known noisy warnings in normal mode ───────────────────────
if not VERBOSE:
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="aiter")
    warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*pad_token_id.*")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── ROCm performance defaults (must be set before any torch import) ──────
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.1")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", "TRUE")

import torch

# ── 内部状态 ──
_model = None
_compile_applied = False


def get_model(model_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"):
    """获取或创建 Qwen3TTSModel 单例，自动启用 torch.compile(reduce-overhead)。"""
    global _model, _compile_applied

    if _model is None:
        from qwen_tts import Qwen3TTSModel

        with _quiet():
            _model = Qwen3TTSModel.from_pretrained(
                model_id, device_map="cuda:0", torch_dtype=torch.bfloat16,
            )

        # torch.compile with reduce-overhead (includes CUDA Graph optimization)
        if not _compile_applied:
            if VERBOSE:
                print("  Applying torch.compile(mode='reduce-overhead')...", end=" ", flush=True)
            with _quiet():
                _model.model = torch.compile(_model.model, mode="reduce-overhead")
            _compile_applied = True

            # Warmup: first run triggers JIT compilation
            if VERBOSE:
                print("warmup...", end=" ", flush=True)
            with _quiet():
                _model.generate_custom_voice(
                    text="你好。", speaker="Vivian", language="chinese",
                )
            if VERBOSE:
                print("done")

    return _model


AVAILABLE_SPEAKERS = [
    "Vivian", "Serena", "Ono_Anna", "Sohee",
    "Uncle_Fu", "Ryan", "Aiden", "Eric", "Dylan",
]

DEFAULT_SPEAKER = "Uncle_Fu"
DEFAULT_INSTRUCT = "narration"
DEFAULT_LANGUAGE = "chinese"


# ─── Inference ──────────────────────────────────────────────────────────

def infer_paragraph(text, speaker=DEFAULT_SPEAKER, instruct=DEFAULT_INSTRUCT,
                    language=DEFAULT_LANGUAGE):
    """生成一段文本的语音。返回 (wav_numpy_1d, sample_rate=24000)。"""
    model = get_model()
    with _quiet():
        audios, sr = model.generate_custom_voice(
            text=text, speaker=speaker, language=language, instruct=instruct,
        )
    wav = np.array(audios[0], dtype=np.float32)
    peak = max(abs(wav.max()), abs(wav.min()), 1e-8)
    wav = np.clip(wav / peak, -1.0, 1.0)
    return wav, sr


def generate_preview(text="你好，欢迎使用语音合成。", speaker=DEFAULT_SPEAKER,
                     instruct=DEFAULT_INSTRUCT, language=DEFAULT_LANGUAGE):
    """生成短试听音频。"""
    return infer_paragraph(text=text, speaker=speaker, instruct=instruct, language=language)


def save_preview_wav(audio, path, sample_rate=24000):
    import soundfile as sf
    sf.write(path, audio, sample_rate)
