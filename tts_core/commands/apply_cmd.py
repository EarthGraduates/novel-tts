"""novel-tts apply — Apply manifest + detail edits to novel.json."""

import os
import re
import sys

from tts_core.config import (
    ensure_novels_dirs,
    manifest_path,
    manifest_detail_path,
    novel_path,
    load_novel,
    save_novel,
    backup_file,
)
from tts_core.parser import generate_manifest


# Mapping from short forms to canonical op names
OP_ALIASES = {
    # Short forms (preferred)
    "K": "keep",
    "M": "merge",
    "S": "split",
    "X": "skip",
    # Legacy long forms (still accepted)
    "keep": "keep",
    "merge": "merge",
    "split": "split",
    "skip": "skip",
}


def _normalize_op(raw):
    """Normalize op code to canonical form. Supports both M/K/S/X and old forms."""
    return OP_ALIASES.get(raw, raw)


def parse_manifest(mpath):
    """Parse manifest.txt → list of {op, id, title}."""
    ops = []
    if not os.path.exists(mpath):
        return ops

    with open(mpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("OP") or line.startswith("---"):
                continue
            parts = line.split(None, 3)
            if len(parts) >= 2:
                op = _normalize_op(parts[0])
                ch_id = parts[1]
                title = parts[3] if len(parts) > 3 else ""
                ops.append({"op": op, "id": ch_id, "title": title})
    return ops


def parse_detail(detail_path):
    """Parse manifest_<ID>.txt → list of SPLIT positions (sentence indices).

    Returns list of sentence positions (0-indexed within the detail) where SPLIT occurs.
    """
    split_positions = []
    if not os.path.exists(detail_path):
        return split_positions

    with open(detail_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find SPLIT markers — they appear between sentences
    # Count sentences to find SPLIT positions
    lines = content.split("\n")
    sent_idx = -1  # will be incremented for each real sentence line
    for line in lines:
        if line.startswith("#"):
            continue  # comment
        stripped = line.strip()
        if stripped.startswith(">>> SPLIT <<<"):
            split_positions.append(sent_idx)
            continue
        if stripped:  # non-empty, non-comment line = sentence text
            sent_idx += 1

    return split_positions


def run(args):
    novels_dir = ensure_novels_dirs()

    # Find novel.json
    novels = [f for f in os.listdir(novels_dir) if f.endswith("_novel.json")]
    if not novels:
        print("❌ 未找到 *_novel.json")
        sys.exit(1)

    if len(novels) == 1:
        book_name = novels[0].replace("_novel.json", "")
    else:
        print("📖 已解析的小说:")
        for i, n in enumerate(novels, 1):
            name = n.replace("_novel.json", "")
            print(f"   [{i}] {name}")
        print()
        choice = input("   选哪个？[1]: ").strip() or "1"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(novels):
                book_name = novels[idx].replace("_novel.json", "")
            else:
                book_name = novels[0].replace("_novel.json", "")
        except ValueError:
            book_name = novels[0].replace("_novel.json", "")
    novel = load_novel(book_name)
    if not novel:
        print("❌ 无法加载 novel.json")
        sys.exit(1)

    mpath = manifest_path(book_name)
    if not os.path.exists(mpath):
        print("❌ 没有 manifest.txt，请先运行 novel-tts parse")
        print("   或: novel-tts manifest 生成章节清单")
        sys.exit(1)

    # Parse manifest ops
    manifest_ops = parse_manifest(mpath)
    if not manifest_ops:
        print("❌ manifest.txt 为空或格式错误")
        sys.exit(1)

    # Collect chapters from novel.json
    toc = novel.get("toc", [])
    all_chapters = []
    for vol in toc:
        if vol.get("type") == "front_matter":
            continue
        for part in vol.get("parts", []):
            for ch in part.get("chapters", []):
                all_chapters.append(ch)

    # Build a map: chapter_id → manifest_op
    op_map = {op["id"]: op for op in manifest_ops}

    # Step 1: Process SPLIT operations from detail files
    split_changes = []  # (chapter_id, split_positions)
    for ch_id in op_map:
        if op_map[ch_id]["op"] == "split":
            detail_path = manifest_detail_path(book_name, ch_id)
            if os.path.exists(detail_path):
                positions = parse_detail(detail_path)
                if positions:
                    split_changes.append((ch_id, positions))
                    print(f"🔧 检测到 SPLIT: {ch_id} @ 位置 {positions}")

    # Step 2: Apply MERGE — combine chapters
    merge_changes = set()
    for ch_id in op_map:
        if op_map[ch_id]["op"] == "merge":
            merge_changes.add(ch_id)

    # Step 3: Apply SKIP (front_matter)
    skip_changes = set()
    for ch_id in op_map:
        if op_map[ch_id]["op"] == "skip":
            skip_changes.add(ch_id)

    changes_made = len(split_changes) + len(merge_changes) + len(skip_changes)

    if changes_made == 0:
        print("✅ 无修改")
        return

    # Backup
    npath = novel_path(book_name)
    backup_file(npath)
    backup_file(mpath)
    print("📦 已备份 novel.json 和 manifest.txt")

    # Apply changes to toc and sentences
    sentences = novel.get("sentences", [])
    new_chapters = []
    merged_into_prev = False

    for i, ch in enumerate(all_chapters):
        ch_id = ch["id"]
        op = op_map.get(ch_id, {}).get("op", "keep")

        if op == "merge":
            # Merge into previous chapter: delete this chapter title, append sentences to prev
            merged_into_prev = True

            # Remove chapter title sentence
            sentences = [
                s for s in sentences
                if not (s["chapter_id"] == ch_id and s["type"] == "chapter_title")
            ]
            # Reassign remaining sentences to previous chapter
            prev_ch = new_chapters[-1] if new_chapters else None
            if prev_ch:
                prev_id = prev_ch["id"]
                for s in sentences:
                    if s["chapter_id"] == ch_id:
                        s["chapter_id"] = prev_id
                # Update paragraphs
                prev_paras = prev_ch["paragraphs"]
                for para in ch.get("paragraphs", []):
                    prev_paras.append(para)
                prev_ch["paragraphs"] = prev_paras
                print(f"🔧 merge: {ch_id} → {prev_id}")
            continue

        if op == "skip":
            # Mark sentences as front_matter
            for s in sentences:
                if s["chapter_id"] == ch_id:
                    s["type"] = "front_matter"
            print(f"🔧 skip: {ch_id} → front_matter")
            continue

        # keep or split
        if ch_id in dict(split_changes):
            positions = dict(split_changes)[ch_id]
            # Split this chapter into N+1 chapters
            new_chs = _split_chapter(ch, sentences, positions)
            new_chapters.extend(new_chs)
            print(f"🔧 split: {ch_id} → {len(new_chs)} 章 ({[c['id'] for c in new_chs]})")
        else:
            new_chapters.append(ch)

    # Step 4: Renumber chapters
    _renumber_chapters(new_chapters, sentences)

    # Step 5: Rebuild TOC
    novel["toc"] = _rebuild_toc(new_chapters, sentences, novel.get("toc", []))

    # Step 6: Save
    save_novel(book_name, novel)
    print(f"✅ 已应用 {changes_made} 处修改")

    # Step 7: Regenerate manifest
    manifest_content = generate_manifest(novel)
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(manifest_content)
    print(f"✅ manifest 已重新生成")

    # Print summary
    print()
    print("📋 章节清单:")
    for vol in novel["toc"]:
        if vol.get("type") == "front_matter":
            continue
        for part in vol.get("parts", []):
            for ch in part.get("chapters", []):
                sr = ch.get("sentence_range", [0, 0])
                cnt = sr[1] - sr[0] + 1 if sr[1] >= sr[0] else 0
                print(f"   {ch['id']}  {ch['title'][:40]:40s} {cnt:4d}句")


def _split_chapter(chapter, sentences, split_positions):
    """Split a chapter at given sentence positions. Returns list of new chapter dicts."""
    ch_id_orig = chapter["id"]
    ch_title = chapter.get("title", "")

    # Get all sentences for this chapter, sorted by order
    ch_sents = sorted(
        [s for s in sentences if s["chapter_id"] == ch_id_orig],
        key=lambda s: s["order"]
    )

    # Remove chapter title from split consideration
    title_sent = None
    content_sents = []
    for s in ch_sents:
        if s["type"] == "chapter_title":
            title_sent = s
        else:
            content_sents.append(s)

    # Split positions are 0-indexed within content_sents
    split_indices = [0] + [p + 1 for p in split_positions] + [len(content_sents)]
    new_chapters = []

    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        if start >= end:
            continue

        sub_sents = content_sents[start:end]
        if not sub_sents:
            continue

        new_ch_id = f"{len(new_chapters) + int(ch_id_orig):04d}" if i > 0 else ch_id_orig

        if i == 0:
            # Keep original chapter ID
            new_title = ch_title + "（上）" if len(split_indices) > 2 else ch_title
            if title_sent:
                title_sent["text"] = new_title
        else:
            new_title = f"{ch_title}（下）" if len(split_indices) == 3 else f"{ch_title}（{i+1}）"
            # Create new chapter title sentence
            max_order = max(s["order"] for s in sentences) if sentences else 0
            new_title_sent = {
                "id": f"{new_ch_id}-000000",
                "order": sub_sents[0]["order"] - 1,  # insert before first sentence
                "chapter_id": new_ch_id,
                "type": "chapter_title",
                "text": new_title,
                "status": "pending",
                "audio_path": "",
            }
            sentences.append(new_title_sent)

        # Reassign sentences to new chapter
        for s in sub_sents:
            s["chapter_id"] = new_ch_id

        # Build paragraph ranges
        para_ranges = []
        for s in sub_sents:
            if "status" in s:
                para_ranges.append([s["order"], s["order"]])  # will be expanded later
        # Simple: just one paragraph for the whole sub-chapter
        para_ranges = [[sub_sents[0]["order"], sub_sents[-1]["order"]]]

        new_chapters.append({
            "id": new_ch_id,
            "title": new_title,
            "status": "pending",
            "audio_path": "",
            "paragraphs": para_ranges,
        })

    return new_chapters


def _renumber_chapters(chapters, sentences):
    """Renumber chapters and sentence IDs after merges/splits."""
    for i, ch in enumerate(chapters):
        new_id = f"{i + 1:04d}"
        old_id = ch["id"]

        if new_id != old_id:
            ch["id"] = new_id
            for s in sentences:
                if s["chapter_id"] == old_id:
                    s["chapter_id"] = new_id

    # Recalculate sentence order globally
    sentences.sort(key=lambda s: s["order"])
    for i, s in enumerate(sentences):
        s["order"] = i + 1

    # Recalculate sentence IDs
    chapter_counters = {}
    for s in sentences:
        ch_id = s["chapter_id"]
        if s["type"] == "chapter_title":
            s["id"] = f"{ch_id}-000000"
            chapter_counters[ch_id] = 1
        else:
            count = chapter_counters.get(ch_id, 0)
            s["id"] = f"{ch_id}-{count:06d}"
            chapter_counters[ch_id] = count + 1


def _rebuild_toc(chapters, sentences, old_toc):
    """Rebuild TOC from new chapters list."""
    # Preserve volume structure from old TOC
    new_ch_list = []
    for ch in chapters:
        ch_id = ch["id"]
        ch_sents = [s for s in sentences if s["chapter_id"] == ch_id]
        if ch_sents:
            min_o = min(s["order"] for s in ch_sents)
            max_o = max(s["order"] for s in ch_sents)
        else:
            min_o = max_o = 0
        new_ch_list.append({
            "id": ch_id,
            "title": ch.get("title", ""),
            "status": ch.get("status", "pending"),
            "audio_path": ch.get("audio_path", ""),
            "sentence_range": [min_o, max_o],
            "paragraphs": ch.get("paragraphs", []),
        })

    # Keep front_matter entries
    new_toc = []
    for vol in old_toc:
        if vol.get("type") == "front_matter":
            new_toc.append(vol)

    new_toc.append({
        "volume": 1,
        "title": "正文",
        "parts": [{
            "part": 1,
            "title": "",
            "chapters": new_ch_list,
        }],
    })

    return new_toc
