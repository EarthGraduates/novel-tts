"""Qwen3-TTS local engine — GPU-accelerated, high-quality.

Built on Qwen3TTSModel with torch.compile and ROCm optimizations.
Returns WAV (NumPy float32 array, 24000 Hz mono).
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

# ── ROCm performance defaults ──────────────────────────────────────────
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.1")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", "TRUE")

import torch

name = "qwen3-tts"
label = "本地 Qwen3-TTS 0.6B（需 GPU，效果更好）"
needs_gpu = True
needs_load = True

SPEAKERS = [
    {"id": "Vivian",    "name": "Vivian",     "gender": "女", "default": False},
    {"id": "Serena",    "name": "Serena",     "gender": "女", "default": False},
    {"id": "Ono_Anna",  "name": "Ono_Anna",   "gender": "女", "default": False},
    {"id": "Sohee",     "name": "Sohee",      "gender": "女", "default": False},
    {"id": "Uncle_Fu",  "name": "Uncle_Fu",   "gender": "男", "default": True},
    {"id": "Ryan",      "name": "Ryan",       "gender": "男", "default": False},
    {"id": "Aiden",     "name": "Aiden",      "gender": "男", "default": False},
    {"id": "Eric",      "name": "Eric",       "gender": "男", "default": False},
    {"id": "Dylan",     "name": "Dylan",      "gender": "男", "default": False},
]

STYLES = [
    {"id": "narration", "name": "旁白", "default": True},
    {"id": "gentle",    "name": "温柔", "default": False},
    {"id": "sad",       "name": "悲伤", "default": False},
    {"id": "angry",     "name": "愤怒", "default": False},
    {"id": "cheerful",  "name": "欢快", "default": False},
    {"id": "serious",   "name": "严肃", "default": False},
]

# ── Internal state ──
_model = None
_compile_applied = False


def load(model_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"):
    """Load Qwen3TTSModel singleton, enable torch.compile."""
    global _model, _compile_applied

    if _model is None:
        from qwen_tts import Qwen3TTSModel

        with _quiet():
            _model = Qwen3TTSModel.from_pretrained(
                model_id, device_map="cuda:0", torch_dtype=torch.bfloat16,
            )

        if not _compile_applied:
            if VERBOSE:
                print("  Applying torch.compile(mode='reduce-overhead')...", end=" ", flush=True)
            with _quiet():
                _model.model = torch.compile(_model.model, mode="reduce-overhead")
            _compile_applied = True

            if VERBOSE:
                print("warmup...", end=" ", flush=True)
            with _quiet():
                _model.generate_custom_voice(
                    text="你好。", speaker="Vivian", language="chinese",
                )
            if VERBOSE:
                print("done")


def infer(text, speaker="Uncle_Fu", style="narration", language="chinese"):
    """Generate audio via Qwen3-TTS. Returns (wav_numpy_1d, sr=24000)."""
    model = _model
    if model is None:
        load()
        model = _model

    with _quiet():
        audios, sr = model.generate_custom_voice(
            text=text, speaker=speaker, language=language, instruct=style,
        )
    wav = np.array(audios[0], dtype=np.float32)
    peak = max(abs(wav.max()), abs(wav.min()), 1e-8)
    wav = np.clip(wav / peak, -1.0, 1.0)
    return wav, sr


def preview(text="你好，欢迎使用语音合成。", speaker="Uncle_Fu",
            style="narration", language="chinese"):
    """Short preview. Same as infer for Qwen3-TTS."""
    return infer(text, speaker, style, language)


# ── Backward-compatible aliases ─────────────────────────────────────────
AVAILABLE_SPEAKERS = [s["id"] for s in SPEAKERS]
DEFAULT_SPEAKER = "Uncle_Fu"
DEFAULT_INSTRUCT = "narration"
DEFAULT_LANGUAGE = "chinese"

get_model = load
infer_paragraph = infer
generate_preview = preview


def save_preview_wav(audio, path, sample_rate=24000):
    import soundfile as sf
    sf.write(path, audio, sample_rate)