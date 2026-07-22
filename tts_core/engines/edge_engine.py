"""Microsoft Edge TTS engine — free, no GPU, cloud-based.

Uses the edge-tts library to access Microsoft's free TTS service.
Returns WAV (NumPy array, 24000 Hz mono) for unified processing.
"""

import asyncio
import io
import numpy as np

name = "edge-tts"
label = "微软 Edge TTS（免费，无需 GPU，速度快）"
needs_gpu = False
needs_load = False

SPEAKERS = [
    {"id": "zh-CN-XiaoxiaoNeural",  "name": "晓晓",  "gender": "女", "default": True},
    {"id": "zh-CN-YunxiNeural",     "name": "云希",  "gender": "男", "default": False},
    {"id": "zh-CN-YunjianNeural",   "name": "云健",  "gender": "男", "default": False},
    {"id": "zh-CN-XiaoyiNeural",    "name": "晓伊",  "gender": "女", "default": False},
    {"id": "zh-CN-YunyangNeural",   "name": "云扬",  "gender": "男", "default": False},
]

STYLES = [
    {"id": "general",     "name": "通用"},
    {"id": "cheerful",    "name": "欢快"},
    {"id": "sad",         "name": "悲伤"},
    {"id": "angry",       "name": "愤怒"},
    {"id": "gentle",      "name": "温柔"},
    {"id": "serious",     "name": "严肃"},
    {"id": "narration",   "name": "旁白"},
]


def load():
    """Edge TTS has no local model to load."""
    pass


def infer(text, speaker="zh-CN-XiaoxiaoNeural", style="general",
          language="zh-CN"):
    """Generate audio via Edge TTS. Returns (wav_numpy_1d, sample_rate=24000)."""
    return _run_async(_infer_async(text, speaker, style, language))


def preview(text="你好，欢迎使用语音合成。", speaker="zh-CN-XiaoxiaoNeural",
            style="general", language="zh-CN"):
    """Short preview. Same as infer for Edge TTS."""
    return infer(text, speaker, style, language)


async def _infer_async(text, speaker, style, language):
    """Async Edge TTS call. Collects MP3 chunks, decodes to WAV."""
    import edge_tts

    communicate = edge_tts.Communicate(
        text=text,
        voice=speaker,
        rate="+0%",
        pitch="+0Hz",
    )

    mp3_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data.extend(chunk["data"])

    # Decode MP3 → NumPy array (24kHz mono)
    return _mp3_to_wav(bytes(mp3_data))


def _mp3_to_wav(mp3_bytes):
    """Decode MP3 bytes to NumPy float32 array at 24000 Hz mono."""
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32")
    # Ensure mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def save_preview_wav(audio, path, sample_rate=24000):
    import soundfile as sf
    sf.write(path, audio, sample_rate)


def _run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already in an event loop — use a subprocess call or thread
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()