"""ChatTTS model loading and speaker management utilities.

All heavy imports (torch, ChatTTS, numpy, soundfile) are lazy — only loaded
when a function that needs them is called, not at module import time.
"""

import os
import sys

# ChatTTS package path (in conda env)
CHATTTS_PACKAGE = os.path.expanduser(
    "~/miniconda3/envs/chattts/lib/python3.10/site-packages"
)
if CHATTTS_PACKAGE not in sys.path:
    sys.path.insert(0, CHATTTS_PACKAGE)

# AMD ROCm environment (from chattts_web.py)
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.1")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

_chat_instance = None
_chatts_imported = False


def _ensure_imports():
    """Lazy-import torch, ChatTTS, numpy, soundfile (heavy deps)."""
    global _chatts_imported
    if not _chatts_imported:
        import torch  # noqa: F811
        import numpy as np  # noqa: F811
        import soundfile  # noqa: F811
        import ChatTTS  # noqa: F811
        _chatts_imported = True


def get_chat():
    """Get or create the ChatTTS Chat instance (loads model on first call)."""
    global _chat_instance
    _ensure_imports()
    import ChatTTS
    if _chat_instance is None:
        _chat_instance = ChatTTS.Chat()
        # Assets are at /home/phoenix/asset/, not ./asset/
        _chat_instance.load(source="custom", compile=False, custom_path="/home/phoenix")
    return _chat_instance


def generate_preset_speakers(n=5, seed=42):
    """Generate n preset speaker embeddings with fixed seed for reproducibility."""
    import torch
    chat = get_chat()
    torch.manual_seed(seed)
    presets = {}
    for i in range(1, n + 1):
        presets[f"preset_{i}"] = chat.sample_random_speaker()
    return presets


def extract_speaker_from_audio(wav_path):
    """Extract speaker embedding from reference audio file."""
    import soundfile as sf
    chat = get_chat()
    wav, sr = sf.read(wav_path)
    spk_emb = chat.sample_audio_speaker(wav)
    return spk_emb


def generate_preview(text="你好，欢迎使用语音合成。", spk_emb=None):
    """Generate a short preview audio and return the numpy array."""
    import torch
    import numpy as np
    from ChatTTS.core import Chat as ChatModule

    chat = get_chat()
    kwargs = {"use_decoder": True}
    if spk_emb is not None:
        kwargs["params_infer_code"] = ChatModule.InferCodeParams(spk_emb=spk_emb)
    wavs = chat.infer([text], **kwargs)
    audio = wavs[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    audio = np.squeeze(audio)
    audio = audio / max(abs(audio.max()), abs(audio.min()), 1e-8)
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def save_preview_wav(audio, path):
    """Save preview audio as WAV file."""
    import soundfile as sf
    sf.write(path, audio, 24000)


def build_infer_params(voice_profile):
    """Build InferCodeParams and RefineTextParams from voice_profile dict.

    Returns (params_refine_text, params_infer_code).
    """
    from ChatTTS.core import Chat as ChatModule

    speed = voice_profile.get("speed", 5)
    oral = voice_profile.get("oral", 3)
    laugh = voice_profile.get("laugh", 0)
    break_val = voice_profile.get("break", 0)

    params_infer_code = ChatModule.InferCodeParams(
        prompt=f"[speed_{speed}]",
    )
    params_refine_text = ChatModule.RefineTextParams(
        prompt=f"[oral_{oral}][laugh_{laugh}][break_{break_val}]",
    )
    return params_refine_text, params_infer_code
