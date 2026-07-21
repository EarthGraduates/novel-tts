"""novel-tts generate — Start audio generation with resume support (Qwen3-TTS)."""

import os
import sys
import time
import csv
from datetime import datetime
from collections import deque
import numpy as np

from tts_core.config import (
    ensure_novels_dirs, load_novel, save_novel, load_config,
)
from tts_core.qwen3tts_utils import (
    get_model, infer_paragraph, AVAILABLE_SPEAKERS,
    DEFAULT_SPEAKER, DEFAULT_INSTRUCT, DEFAULT_LANGUAGE,
)

OUTPUT_BASE = "novels/output"
ETA_WINDOW = 10  # number of recent segments for ETA calculation


def _init_log(log_path):
    """Create log file with CSV header if it doesn't exist."""
    if not os.path.exists(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "chapter_id", "start_order",
                             "end_order", "chars", "duration_s", "status"])


def _log_segment(log_path, ch_id, start_order, end_order, chars, duration_s, status):
    """Append one segment result to the generate log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, ch_id, start_order, end_order,
                         chars, f"{duration_s:.1f}", status])


def _read_log_durations(log_path, window=ETA_WINDOW):
    """Read the last N successful durations from the log, returns list of floats."""
    if not os.path.exists(log_path):
        return []
    durations = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 7 and row[6] == "ok":
                try:
                    durations.append(float(row[5]))
                except ValueError:
                    pass
    return durations[-window:]


def _format_eta(remaining, avg_duration_s):
    """Format ETA string from remaining count and average duration."""
    total_s = remaining * avg_duration_s
    if total_s < 60:
        return f"{total_s:.0f}s"
    elif total_s < 3600:
        return f"{total_s / 60:.0f}m{total_s % 60:.0f}s"
    else:
        h = int(total_s / 3600)
        m = int((total_s % 3600) / 60)
        return f"{h}h{m}m"


def run(args):
    novels_dir = ensure_novels_dirs()

    # Parse args: [chapter_filter] or [book_name] [chapter_filter]
    book_name = None
    chapter_filter = "X"  # default: all

    for a in (args or []):
        if a == "X" or (len(a) == 4 and a.isdigit()):
            chapter_filter = a
        else:
            book_name = a.replace("_novel.json", "").replace(".json", "")

    # Auto-detect book_name if not specified
    if not book_name:
        novels = [f for f in os.listdir(novels_dir) if f.endswith("_novel.json")]
        if not novels:
            print("❌ 未找到 *_novel.json，请先运行 novel-tts parse")
            sys.exit(1)
        if len(novels) > 1:
            print("多个 novel.json，请指定: novel-tts generate <书名> [章节]")
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

    sentences = novel.get("sentences", [])
    all_para_starts = [s for s in sentences if "status" in s]
    toc = novel.get("toc", [])

    # ── Chapter filter ──
    if chapter_filter != "X":
        filtered = []
        matching = []
        for vol in toc:
            if vol.get("type") == "front_matter":
                continue
            for part in vol.get("parts", []):
                m = [ch for ch in part.get("chapters", []) if ch["id"] == chapter_filter]
                if m:
                    matching = m
                    filtered.append({
                        "volume": vol.get("volume", 1),
                        "title": vol.get("title", ""),
                        "parts": [{"part": part.get("part", 1), "title": part.get("title", ""), "chapters": m}],
                    })
        if not filtered:
            all_ids = []
            for vol in toc:
                if vol.get("type") == "front_matter":
                    continue
                for part in vol.get("parts", []):
                    all_ids.extend(ch["id"] for ch in part.get("chapters", []))
            print(f"❌ 未找到章节 {chapter_filter}")
            print(f"   可用章节: {all_ids[0]} ~ {all_ids[-1]}" if all_ids else "")
            sys.exit(1)
        toc = filtered
        # Only count paragraphs in this chapter
        ch_sentence_orders = set()
        for ch in matching:
            for para_range in ch.get("paragraphs", []):
                ch_sentence_orders.update(range(para_range[0], para_range[1] + 1))
        para_starts = [s for s in all_para_starts if s["order"] in ch_sentence_orders]
        para_count_ch = sum(len(ch.get("paragraphs", [])) for ch in matching)
        print(f"📑 只生成章节: {chapter_filter}（{para_count_ch} 段）")
    else:
        para_starts = all_para_starts

    # Count progress — reconcile JSON status with actual files on disk
    total = len(para_starts)
    output_dir = os.path.join(os.getcwd(), OUTPUT_BASE, book_name)
    os.makedirs(output_dir, exist_ok=True)

    # Init generate log
    log_path = os.path.join(output_dir, "generate.log")
    _init_log(log_path)

    # Scan existing WAV files
    existing_wavs = set()
    if os.path.isdir(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".wav"):
                    existing_wavs.add(os.path.join(root, f))

    done = 0; fixed_file_missing = 0; fixed_file_found = 0
    for s in para_starts:
        expected_path = os.path.join(output_dir, s.get("audio_path", "")) if s.get("audio_path") else ""
        file_exists = expected_path in existing_wavs if expected_path else False

        if s["status"] == "done":
            if file_exists:
                done += 1
            else:
                s["status"] = "pending"
                s["audio_path"] = ""
                fixed_file_missing += 1
        elif s["status"] in ("pending", "error", "failed"):
            if file_exists:
                s["status"] = "done"
                done += 1
                fixed_file_found += 1

    errors = sum(1 for s in para_starts if s["status"] == "error")
    failed = sum(1 for s in para_starts if s["status"] == "failed")
    pending = total - done - errors - failed

    # Fix detection summary
    fix_msgs = []
    if fixed_file_missing:
        fix_msgs.append(f"{fixed_file_missing} 段文件丢失 → 重新生成")
    if fixed_file_found:
        fix_msgs.append(f"{fixed_file_found} 段文件存在 → 标记完成")
    if fix_msgs:
        save_novel(book_name, novel)

    print()
    print(f"📖 {novel.get('novel_title', book_name)}")
    print(f"🎤 引擎: Qwen3-TTS 0.6B")
    print(f"🔊 Speaker: {speaker}  |  🎭 风格: {instruct}")
    print()
    print(f"📊 目标: {total} 段")
    print(f"   已完成: {done} 段 ({100*done//max(total,1)}%)")
    if fix_msgs:
        for m in fix_msgs:
            print(f"   🔧 {m}")
    if errors:
        print(f"   错误: {errors} 段")
    if failed:
        print(f"   失败: {failed} 段")
    if pending:
        print(f"   待处理: {pending} 段")

    if pending == 0:
        print(f"\n✅ 全部完成！")
        return

    if done > 0:
        print(f"\n   [1] 断点续传（推荐）  [2] 重新生成")
        if input("   选哪个？[1]: ").strip() == "2":
            for s in para_starts:
                if s["status"] == "done":
                    continue  # keep done segments
                s["status"] = "pending"
                s["audio_path"] = ""
            save_novel(book_name, novel)
            done = sum(1 for s in para_starts if s["status"] == "done")
            errors = failed = 0
            print(f"   已重置。保持 {done} 段已完成")
    else:
        print()

    # ── Generation loop ──
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

                    # ETA
                    recent = _read_log_durations(log_path)
                    eta_str = ""
                    if recent and pending > 0:
                        avg_s = sum(recent) / len(recent)
                        remaining = pending - done
                        eta_str = f" | ETA {_format_eta(remaining, avg_s)}"

                    print(f"[{para_count}/{total}] {ch_id}/段[{start_order}-{end_order}]{eta_str}",
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
                        _log_segment(log_path, ch_id, start_order, end_order,
                                     len(text), dur, "ok")
                        print(f"✓ {dur:.1f}s")
                    except Exception as e:
                        _log_segment(log_path, ch_id, start_order, end_order,
                                     len(text), 0, "error_retry")
                        print(f"✗ {str(e)[:80]}")
                        time.sleep(3)
                        try:
                            dur = _gen_and_save()
                            _log_segment(log_path, ch_id, start_order, end_order,
                                         len(text), dur, "ok")
                            print("✓")
                        except Exception:
                            para_s["status"] = "error"
                            para_s["audio_path"] = ""
                            save_novel(book_name, novel)
                            error_list.append((ch_id, (start_order, end_order)))
                            _log_segment(log_path, ch_id, start_order, end_order,
                                         len(text), 0, "error")
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
                _log_segment(log_path, ch_id, start_order, end_order,
                             len(text), len(wav) / sr, "ok_retry")
                print("  ✓")
            except Exception:
                para_s["status"] = "failed"
                save_novel(book_name, novel)
                _log_segment(log_path, ch_id, start_order, end_order,
                             len(text), 0, "failed")
                print("  ✗ → failed")

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
    print(f"📋 生成日志: {log_path}")
    if fe > 0 or ff > 0:
        print(f"💡 运行 novel-tts generate 进行断点续传")


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
