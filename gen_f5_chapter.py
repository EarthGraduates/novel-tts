"""Generate chapter 0001 with F5-TTS for quality comparison with ChatTTS."""
import time
import json
import sys
import numpy as np
import torch
import soundfile as sf

# ---- Monkey-patch torchaudio for AMD ROCm (same as test_f5_tts.py) ----
import torchaudio

def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **kwargs):
    data, sample_rate = sf.read(uri, dtype="float32", always_2d=True)
    data = torch.from_numpy(data)
    if frame_offset > 0:
        data = data[frame_offset:]
    if num_frames > 0:
        data = data[:num_frames]
    if channels_first:
        data = data.transpose(0, 1)
    return data, sample_rate

def _sf_save(uri, src, sample_rate, channels_first=True, **kwargs):
    if isinstance(src, torch.Tensor):
        src = src.detach().cpu()
        if src.ndim == 2 and channels_first:
            src = src.transpose(0, 1)
        src = src.numpy()
    sf.write(uri, src, sample_rate)

torchaudio.load = _sf_load
torchaudio.save = _sf_save
torchaudio.load_with_torchcodec = _sf_load
torchaudio.save_with_torchcodec = _sf_save

# ---- Now safe to import ----
from f5_tts.api import F5TTS
from importlib.resources import files

# ---- Config ----
NOVEL_JSON = "novels/大力金刚掌-茅山后裔_novel.json"
CHAPTER_ID = "0001"
OUTPUT_DIR = f"novels/output/大力金刚掌-茅山后裔/f5_{CHAPTER_ID}"
REF_AUDIO = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_zh.wav"))
REF_TEXT = "有的人称呼我为大自然，也有人叫我大自然母亲。"

# ---- Load chapter text ----
with open(NOVEL_JSON) as f:
    novel = json.load(f)

toc = novel["toc"]
sentences = novel["sentences"]

chapter = None
for vol in toc:
    if vol.get("type") == "front_matter":
        continue
    for part in vol.get("parts", []):
        for ch in part.get("chapters", []):
            if ch["id"] == CHAPTER_ID:
                chapter = ch
                break

if chapter is None:
    print(f"❌ Chapter {CHAPTER_ID} not found")
    sys.exit(1)

print(f"📖 {chapter['title']}")
print(f"   段落数: {len(chapter['paragraphs'])}")
print(f"   参考音频: {REF_AUDIO}")
print()

# ---- Init model ----
print("⏳ Loading F5TTS_v1_Base...", end=" ", flush=True)
start = time.time()
f5tts = F5TTS(model="F5TTS_v1_Base")
print(f"✓ ({time.time() - start:.1f}s)")
print()

# ---- Generate each paragraph ----
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

para_wavs = []
total = len(chapter["paragraphs"])

for idx, pr in enumerate(chapter["paragraphs"]):
    start_order, end_order = pr
    para_sents = [s for s in sentences if start_order <= s["order"] <= end_order]
    para_sents.sort(key=lambda s: s["order"])
    text = "".join(s["text"] for s in para_sents)

    print(f"[{idx+1}/{total}] 段[{start_order}-{end_order}] {len(text)} chars — ", end="", flush=True)

    try:
        t0 = time.time()
        wav, sr, _ = f5tts.infer(
            ref_file=REF_AUDIO,
            ref_text=REF_TEXT,
            gen_text=text,
            seed=42,
            nfe_step=32,
        )
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()
        elif isinstance(wav, np.ndarray):
            wav = wav.squeeze()
        else:
            wav = np.squeeze(np.array(wav))

        # Save paragraph
        para_path = os.path.join(OUTPUT_DIR, f"p_{CHAPTER_ID}_{start_order}.wav")
        sf.write(para_path, wav, sr)
        para_wavs.append(para_path)

        dur = len(wav) / sr
        rt = time.time() - t0
        rtf = rt / dur if dur > 0 else 0
        print(f"✓ {dur:.1f}s | RTF={rtf:.2f}")

    except Exception as e:
        print(f"✗ {str(e)[:80]}")
        torch.cuda.empty_cache()
        time.sleep(2)

    torch.cuda.empty_cache()

# ---- Concatenate ----
print()
print("🔧 Concatenating...", end=" ", flush=True)

all_data = []
for path in para_wavs:
    data, sr = sf.read(path)
    all_data.append(data)

combined = np.concatenate(all_data)
chapter_path = os.path.join(OUTPUT_DIR, "chapter.wav")
sf.write(chapter_path, combined, sr)

total_dur = len(combined) / sr
print(f"✓ {total_dur:.1f}s total")
print(f"📁 {chapter_path}")
print()

# ---- Comparison ----
chattts_path = f"novels/output/大力金刚掌-茅山后裔/0001/chapter.wav"
if os.path.exists(chattts_path):
    chattts_data, _ = sf.read(chattts_path)
    chattts_dur = len(chattts_data) / 24000
    print(f"📊 Comparison:")
    print(f"   ChatTTS: {chattts_dur:.1f}s  →  {chattts_path}")
    print(f"   F5-TTS:  {total_dur:.1f}s  →  {chapter_path}")
else:
    print("📝 ChatTTS chapter.wav not found for comparison")

print()
print("✅ Done!")
