"""tts generate — Start audio generation with resume support (Qwen3-TTS)."""

import os
import sys
import time
import numpy as np

from tts_core.config import (
    ensure_novels_dirs, load_novel, save_novel, load_config,
)
from tts_core.qwen3tts_utils import (
    get_model, infer_paragraph, AVAILABLE_SPEAKERS,
    DEFAULT_SPEAKER, DEFAULT_INSTRUCT, DEFAULT_LANGUAGE,
)

OUTPUT_BASE = "novels/output"


def run(args):
    novels_dir = ensure_novels_dirs()

    # Find novel.json
    if args:
        book_name = args[0].replace("_novel.json", "").replace(".json", "")
    else:
        novels = [f for f in os.listdir(novels_dir) if f.endswith("_novel.json")]
        if not novels:
            print("❌ 未找到 *_novel.json，请先运行 tts parse")
            sys.exit(1)
        if len(novels) > 1:
            print("多个 novel.json，请指定: tts generate <书名>")
            sys.exit(1)
        book_name = novels[0].replace("_novel.json", "")

    novel = load_novel(book_name)
    if not novel:
        print(f"❌ 无法加载 novel.json")
        sys.exit(1)

    # Load voice config
    config = load_config(book_name)
    voice = config.get("voice_profile", {}) if config else novel.get("voice_profile", {})

    speaker = voice.get("speaker", DEFAULT_SPEAKER)
    instruct = voice.get("instruct", DEFAULT_INSTRUCT)
    language = voice.get("language", DEFAULT_LANGUAGE)

    if speaker not in AVAILABLE_SPEAKERS:
        print(f"⚠️  未知 speaker '{speaker}'，回退到 {DEFAULT_SPEAKER}")
        speaker = DEFAULT_SPEAKER

    # Load model (includes torch.compile + warmup)
    print("⏳ 正在加载 Qwen3-TTS 模型...", flush=True)
    get_model()

    # Count progress
    sentences = novel.get("sentences", [])
    para_starts = [s for s in sentences if "status" in s]
    total = len(para_starts)
    done = sum(1 for s in para_starts if s["status"] == "done")
    errors = sum(1 for s in para_starts if s["status"] == "error")
    failed = sum(1 for s in para_starts if s["status"] == "failed")
    pending = total - done - errors - failed

    print()
    print(f"📖 {novel.get('novel_title', book_name)}")
    print(f"🎤 引擎: Qwen3-TTS 0.6B")
    print(f"🔊 Speaker: {speaker}  |  🎭 风格: {instruct}")

    if done > 0 or errors > 0 or failed > 0:
        print(f"\n📊 已有进度: 完成 {done}/{total} | 错误 {errors} | 失败 {failed} | 待处理 {pending}")
        print("\n   [1] 断点续传（推荐）  [2] 重新生成")
        if input("   选哪个？[1]: ").strip() == "2":
            for s in para_starts:
                s["status"] = "pending"
                s["audio_path"] = ""
            save_novel(book_name, novel)
            done, errors, failed = 0, 0, 0
            print("   已重置")
    else:
        print(f"   共 {total} 段待生成\n")

    output_dir = os.path.join(os.getcwd(), OUTPUT_BASE, book_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── Generation loop ──
    toc = novel.get("toc", [])
    para_count = 0
    error_list = []

    for vol in toc:
        if vol.get("type") == "front_matter":
            continue
        for part in vol.get("parts", []):
            for ch in part.get("chapters", []):
                ch_id = ch["id"]
                ch_out = os.path.join(output_dir, ch_id)
                os.makedirs(ch_out, exist_ok=True)

                for start_order, end_order in ch.get("paragraphs", []):
                    para_count += 1

                    # Find paragraph-start sentence
                    para_s = next((s for s in sentences if s["order"] == start_order and "status" in s), None)
                    if para_s is None:
                        continue

                    # Skip done
                    if para_s["status"] == "done":
                        p = para_s.get("audio_path", "")
                        if os.path.exists(os.path.join(output_dir, p)) if p else False:
                            continue

                    # Build text
                    ps = sorted([s for s in sentences if start_order <= s["order"] <= end_order],
                                key=lambda s: s["order"])
                    text = "".join(s["text"] for s in ps)

                    print(f"[{para_count}/{total}] {ch_id}/段[{start_order}-{end_order}]",
                          end=" ", flush=True)

                    def _gen_and_save():
                        wav, sr = infer_paragraph(text, speaker=speaker,
                                                  instruct=instruct, language=language)
                        wav_path = f"{ch_id}/p_{ch_id}_{start_order}.wav"
                        _save_wav(wav, os.path.join(output_dir, wav_path), sr)
                        para_s["status"] = "done"
                        para_s["audio_path"] = wav_path
                        save_novel(book_name, novel)
                        return len(wav) / sr

                    try:
                        dur = _gen_and_save()
                        print(f"✓ {dur:.1f}s")
                    except Exception as e:
                        print(f"✗ {str(e)[:80]}")
                        time.sleep(3)
                        try:
                            dur = _gen_and_save()
                            print("✓")
                        except Exception:
                            para_s["status"] = "error"
                            para_s["audio_path"] = ""
                            save_novel(book_name, novel)
                            error_list.append((ch_id, (start_order, end_order)))
                            print("✗ → error")

    # ── Retry errors ──
    if error_list:
        print(f"\n🔄 重试 {len(error_list)} 个错误段...")
        for ch_id, (start_order, end_order) in error_list:
            para_s = next((s for s in sentences if s["order"] == start_order and "status" in s), None)
            if para_s is None:
                continue
            ps = sorted([s for s in sentences if start_order <= s["order"] <= end_order],
                        key=lambda s: s["order"])
            text = "".join(s["text"] for s in ps)
            print(f"  {ch_id}/段[{start_order}-{end_order}]", end=" ", flush=True)
            try:
                wav, sr = infer_paragraph(text, speaker=speaker, instruct=instruct, language=language)
                wav_path = f"{ch_id}/p_{ch_id}_{start_order}.wav"
                _save_wav(wav, os.path.join(output_dir, wav_path), sr)
                para_s["status"] = "done"
                para_s["audio_path"] = wav_path
                save_novel(book_name, novel)
                print("✓")
            except Exception:
                para_s["status"] = "failed"
                save_novel(book_name, novel)
                print("✗ → failed")

    # ── Chapter concat ──
    print("\n🔧 章节拼接中...")
    for vol in toc:
        if vol.get("type") == "front_matter":
            continue
        for part in vol.get("parts", []):
            for ch in part.get("chapters", []):
                ch_id = ch["id"]
                para_wavs = []
                ok = True
                for start_order, _ in ch.get("paragraphs", []):
                    para_s = next((s for s in sentences if s["order"] == start_order and "status" in s), None)
                    if para_s and para_s["status"] == "done":
                        p = para_s.get("audio_path", "")
                        abs_p = os.path.join(output_dir, p) if p else ""
                        if os.path.exists(abs_p):
                            para_wavs.append(abs_p)
                        else:
                            ok = False
                    else:
                        ok = False

                if para_wavs and ok:
                    para_wavs.sort(key=lambda p: int(os.path.basename(p).rsplit("_", 1)[1].replace(".wav", "")))
                    _concat_wavs(para_wavs, os.path.join(output_dir, ch_id, "chapter.wav"))
                    ch["status"] = "done"
                    ch["audio_path"] = f"{ch_id}/chapter.wav"
                    save_novel(book_name, novel)
                    print(f"  ✓ {ch_id} → {len(para_wavs)} 段")

    # ── Summary ──
    fd = sum(1 for s in sentences if s.get("status") == "done")
    fe = sum(1 for s in sentences if s.get("status") == "error")
    ff = sum(1 for s in sentences if s.get("status") == "failed")
    print(f"\n{'═'*30}")
    print(f"✅ 完成 {fd}/{total}  |  ❌ 错误 {fe}  |  ⛔ 失败 {ff}")
    print(f"📁 {output_dir}")


def _save_wav(wav, path, sample_rate=24000):
    import soundfile as sf
    sf.write(path, wav, sample_rate)


def _concat_wavs(wav_paths, output_path):
    import soundfile as sf
    data, sr = [], None
    for p in wav_paths:
        d, s = sf.read(p)
        if sr is None:
            sr = s
        data.append(d)
    sf.write(output_path, np.concatenate(data), sr)
