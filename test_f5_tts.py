"""Test F5-TTS Chinese speech synthesis.
Monkey-patches torchaudio to use soundfile backend,
since torchcodec requires NVIDIA libnvrtc.so.13 on AMD.
"""
import time
import torch
import soundfile as sf

# ---- Monkey-patch torchaudio BEFORE importing f5_tts ----
# torchaudio 2.11 forces torchcodec which needs libnvrtc.so.13 (NVIDIA-only)
# Replace with soundfile-based load/save for AMD ROCm compatibility
import torchaudio


def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **kwargs):
    data, sample_rate = sf.read(uri, dtype="float32", always_2d=True)
    data = torch.from_numpy(data)  # [time, channel]
    if frame_offset > 0:
        data = data[frame_offset:]
    if num_frames > 0:
        data = data[:num_frames]
    if channels_first:
        data = data.transpose(0, 1)  # [channel, time]
    return data, sample_rate


def _sf_save(uri, src, sample_rate, channels_first=True, **kwargs):
    if isinstance(src, torch.Tensor):
        src = src.detach().cpu()
        if src.ndim == 2 and channels_first:
            src = src.transpose(0, 1)  # [channel, time] -> [time, channel]
        src = src.numpy()
    sf.write(uri, src, sample_rate)


torchaudio.load = _sf_load
torchaudio.save = _sf_save
# Also patch the explicit torchcodec functions
torchaudio.load_with_torchcodec = _sf_load
torchaudio.save_with_torchcodec = _sf_save

# Now safe to import F5-TTS
from f5_tts.api import F5TTS
import f5_tts
from importlib.resources import files

ref_zh = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_zh.wav"))

print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"Reference audio: {ref_zh}")

# Test Chinese sentences
test_texts = [
    ("你好，欢迎使用F5-TTS语音合成系统。", "test_hello"),
    ("今天天气真好，我们一起去公园散步吧。", "test_weather"),
    ("人工智能技术正在改变我们的生活方式，语音合成是其中一个重要的应用领域，它在教育、娱乐、无障碍辅助等方面发挥着越来越大的作用。", "test_ai"),
]

print("\nInitializing F5TTS model...")
start = time.time()
f5tts = F5TTS(model="F5TTS_v1_Base")
print(f"Model loaded in {time.time() - start:.1f}s")

# Provide explicit ref_text to skip auto-transcription
# (auto-transcription also uses torchcodec)
ref_text_zh = "有的人称呼我为大自然，也有人叫我大自然母亲。"

for text, label in test_texts:
    output_file = f"/home/phoenix/ClaudeProjects/TTS/f5_tts_{label}.wav"
    print(f"\n--- {label}: {text} ---")

    start = time.time()
    wav, sr, spec = f5tts.infer(
        ref_file=ref_zh,
        ref_text=ref_text_zh,
        gen_text=text,
        file_wave=output_file,
        seed=42,
    )
    duration = len(wav) / sr
    rt_ms = (time.time() - start) * 1000
    rtf = (time.time() - start) / duration
    print(f"Output: {output_file}")
    print(f"Duration: {duration:.2f}s | Inference: {rt_ms:.0f}ms | RTF: {rtf:.4f}")

print("\n✅ All tests done!")
