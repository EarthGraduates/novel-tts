"""Text parsing engine: txt → novel JSON structure."""

import re
import os


# ── Encoding detection ──────────────────────────────────────────────

def detect_encoding(filepath):
    """Detect file encoding. Returns 'gbk', 'utf-8', or 'gb18030'."""
    # Try UTF-8 first (has BOM or is valid UTF-8)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            f.read(1024)
        return "utf-8"
    except (UnicodeDecodeError, UnicodeError):
        pass

    # Try GBK (most common for Chinese novels)
    try:
        with open(filepath, "r", encoding="gbk") as f:
            f.read(1024)
        return "gbk"
    except (UnicodeDecodeError, UnicodeError):
        pass

    # Fallback to gb18030
    try:
        with open(filepath, "r", encoding="gb18030") as f:
            f.read(1024)
        return "gb18030"
    except (UnicodeDecodeError, UnicodeError):
        pass

    return "utf-8"  # last resort


def read_file(filepath):
    """Read file with auto-detected encoding. Returns list of lines."""
    encoding = detect_encoding(filepath)
    with open(filepath, "r", encoding=encoding) as f:
        lines = f.readlines()
    return lines, encoding


# ── Chapter detection ───────────────────────────────────────────────

# Chinese chapter patterns, ordered by specificity
CHAPTER_PATTERNS = [
    # 第X章 / 第XX章 / 第XXX章 (Chinese numerals + Arabic numerals)
    #   (?<!》) : 前面不能是 》— 排除 "详见《传国宝玺》第五十章" 这类书引
    #   No ^ anchor — chapter titles may have book-name prefix (e.g. 茅山后裔第01章)
    #   Length check in is_chapter_title() also filters body-text false positives
    re.compile(
        r"(?<!》)第[零一二三四五六七八九十百千\d]+[章回节集]"
        r"(?:\s|　|[：:])?"
        r".*$"
    ),
    # Special: 楔子 / 序章 / 尾声 / 后记 / 番外 / 前言 / 引子
    #   Must be at line start (or after whitespace).
    #   "作品相关 序" is handled by preprocess_novel.py — cleaned to just "序"
    re.compile(
        r"(?:^[\s　]*楔子|^[\s　]*序章?|^[\s　]*前言|^[\s　]*引[子言]"
        r"|^[\s　]*尾声|^[\s　]*后记|^[\s　]*番外[篇]?"
        r"|^[\s　]*终章|^[\s　]*大结局)"
        r"(?:\s|　|[：:])?"
        r".*$"
    ),
    # Chapter X (English)
    re.compile(
        r"Chapter\s*\d+",
        re.IGNORECASE
    ),
]

# Volume/part patterns (separate from chapter)
VOLUME_PATTERNS = [
    re.compile(
        r"^[\s　]*"
        r"(?:第[零一二三四五六七八九十百千\d]+[卷部])"
        r"(?:\s|　|[·•])?"
        r".*$"
    ),
]

FRONT_MATTER_PATTERNS = [
    # Pure reference material — not chapters, should not be read aloud
    re.compile(r"作品相关|人物[志表]|角色介绍|设定集|内容简介"),
]


# Max length of a chapter title line in characters.
# Lines longer than this are almost certainly body text, not titles.
MAX_CHAPTER_TITLE_LEN = 50


def is_chapter_title(line, max_len=MAX_CHAPTER_TITLE_LEN):
    """Check if a line contains a chapter title pattern.

    Uses length ≤ max_len to filter false positives (e.g. "详见第一章" in body text
    is typically part of a much longer line). No ^ anchor — chapter titles may have
    a book-name prefix like "茅山后裔第01章".
    """
    stripped = line.strip()
    if len(stripped) > max_len:
        return False
    for pat in CHAPTER_PATTERNS:
        if pat.search(line):
            return True
    return False


def is_volume_title(line):
    """Check if a line matches a volume/part title pattern (standalone volume header)."""
    for pat in VOLUME_PATTERNS:
        if pat.search(line):
            return True
    return False


def is_volume_only(line):
    """Check if a line is ONLY a volume/part header (no chapter number)."""
    # Match: 第X卷 或 第X部, but does NOT contain 第X章
    has_volume = any(
        re.search(r"第[零一二三四五六七八九十百千\d]+[卷部]", line)
        for _ in [1]  # just once
    )
    # Actually let's do it simpler
    has_volume = bool(re.search(r"第[零一二三四五六七八九十百千\d]+[卷部]", line))
    has_chapter = bool(re.search(r"第[零一二三四五六七八九十百千\d]+[章回节集]", line))
    return has_volume and not has_chapter


def is_front_matter_title(line):
    """Check if a line matches front matter patterns."""
    for pat in FRONT_MATTER_PATTERNS:
        if pat.search(line):
            return True
    return False


def has_structural_header(line):
    """Check if a line contains any structural marker (chapter, volume, etc)."""
    return (
        is_chapter_title(line)
        or is_volume_title(line)
        or is_front_matter_title(line)
    )


# ── Sentence splitting ──────────────────────────────────────────────

def split_sentences(text, max_chars=300):
    """Split text into sentences by Chinese punctuation.

    First split by sentence-ending punctuation 。！？…
    Then, if any sentence exceeds max_chars, split again at ，；： within it.
    """
    # Primary split: sentence-ending punctuation
    parts = re.split(r"(?<=[。！？…])", text)

    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # If too long, secondary split at clause-level punctuation
        if len(part) > max_chars:
            sub_parts = re.split(r"(?<=[，；：])", part)
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    sentences.append(sp)
        else:
            sentences.append(part)

    return sentences


# ── Paragraph detection ─────────────────────────────────────────────

def is_blank_line(line):
    """Check if a line is effectively blank."""
    stripped = line.strip().replace("　", "").replace(" ", "")
    return len(stripped) == 0


def is_character_list_line(line):
    """Check if line looks like a character list entry: 　名字　description"""
    return bool(re.match(r"^[　\s]*[^　\s]+[　\s]{2,}[^　\s]", line))


# ── Main parsing pipeline ───────────────────────────────────────────

def parse(filepath):
    """Parse a novel .txt file into novel JSON structure.

    Returns (novel_dict, encoding).
    """
    lines, encoding = read_file(filepath)
    book_name = os.path.splitext(os.path.basename(filepath))[0]

    # Step 1: Clean lines
    cleaned = []
    for line in lines:
        line = line.strip("\r\n")
        # Normalize full-width spaces to regular spaces for processing
        cleaned.append(line)

    # Step 2: Chapter boundary detection
    # First pass: find all lines that are chapter starts
    chapter_starts = []  # list of (line_index, line_text)
    for i, line in enumerate(cleaned):
        if is_chapter_title(line) and not is_front_matter_title(line):
            chapter_starts.append((i, line.strip()))

    # Filter out duplicates (same title, consecutive lines)
    filtered_starts = []
    for idx, (line_idx, title) in enumerate(chapter_starts):
        if idx == 0:
            filtered_starts.append((line_idx, title))
        else:
            prev_title = chapter_starts[idx - 1][1]
            # Skip if same title appeared recently (within 3 lines)
            if title == prev_title and line_idx - chapter_starts[idx - 1][0] <= 3:
                continue
            filtered_starts.append((line_idx, title))

    # Determine front matter boundary: find the "real" chapter start
    # Skip lines that look like front matter (作品相关, 人物表, 序)
    first_real_chapter = 0
    for i, (line_idx, title) in enumerate(filtered_starts):
        if not is_front_matter_title(title) and "作品相关" not in title:
            first_real_chapter = i
            break

    # Build segments
    segments = []
    if first_real_chapter > 0:
        # Everything before first real chapter is front matter
        fm_end = filtered_starts[first_real_chapter][0]
        segments.append(("front_matter", 0, fm_end))

    for i in range(first_real_chapter, len(filtered_starts)):
        line_idx, title = filtered_starts[i]
        next_idx = (
            filtered_starts[i + 1][0] if i + 1 < len(filtered_starts) else len(cleaned)
        )
        segments.append(("chapter", line_idx, next_idx))

    if not segments:
        segments = [("chapter", 0, len(cleaned))]

    # Step 3: Parse chapters from segments
    chapters = []
    all_sentences = []
    global_order = 0
    chapter_counter = 0
    front_matter_sentences_start = None
    front_matter_sentences_end = None

    for seg_idx, (seg_type, seg_start, seg_end) in enumerate(segments):
        if seg_type == "front_matter":
            # Parse front matter into sentences, all marked as front_matter
            seg_lines = cleaned[seg_start:seg_end]
            fm_start_order = global_order + 1

            # Collect non-blank, non-character-list lines as paragraphs
            current_para_lines = []
            for line in seg_lines:
                if is_blank_line(line) or is_character_list_line(line):
                    if current_para_lines:
                        para_text = "".join(current_para_lines).strip()
                        if para_text and len(para_text) > 3:
                            sentences = split_sentences(para_text)
                            para_sents = []
                            for s in sentences:
                                global_order += 1
                                sent = {
                                    "id": "0000-000000",
                                    "order": global_order,
                                    "chapter_id": "0000",
                                    "type": "front_matter",
                                    "text": s,
                                }
                                if not para_sents:
                                    sent["status"] = "pending"
                                    sent["audio_path"] = ""
                                para_sents.append(sent)
                            all_sentences.extend(para_sents)
                        current_para_lines = []
                else:
                    current_para_lines.append(line.strip())

            # Don't forget last paragraph
            if current_para_lines:
                para_text = "".join(current_para_lines).strip()
                if para_text and len(para_text) > 3:
                    sentences = split_sentences(para_text)
                    para_sents = []
                    for s in sentences:
                        global_order += 1
                        sent = {
                            "id": "0000-000000",
                            "order": global_order,
                            "chapter_id": "0000",
                            "type": "front_matter",
                            "text": s,
                        }
                        if not para_sents:
                            sent["status"] = "pending"
                            sent["audio_path"] = ""
                        para_sents.append(sent)
                    all_sentences.extend(para_sents)

            fm_end_order = global_order
            front_matter_sentences_start = fm_start_order
            front_matter_sentences_end = fm_end_order
            continue

        if seg_type != "chapter":
            continue

        chapter_counter += 1
        seg_lines = cleaned[seg_start:seg_end]
        ch_id = f"{chapter_counter:04d}"

        # First line is chapter title
        ch_title_line = seg_lines[0].strip()
        if not ch_title_line:
            ch_title_line = f"第{chapter_counter}章"

        # Chapter title sentence
        global_order += 1
        title_sent = {
            "id": f"{ch_id}-000000",
            "order": global_order,
            "chapter_id": ch_id,
            "type": "chapter_title",
            "text": ch_title_line,
            "status": "pending",
            "audio_path": "",
        }
        all_sentences.append(title_sent)
        ch_para_ranges = [[global_order, global_order]]

        # Parse body paragraphs
        body_lines = seg_lines[1:]
        current_para_lines = []
        for line in body_lines:
            if is_blank_line(line):
                if current_para_lines:
                    para_text = "".join(current_para_lines).strip()
                    if para_text and not is_volume_title(para_text):
                        _add_paragraph(
                            para_text, ch_id, all_sentences, ch_para_ranges, global_order
                        )
                        global_order = all_sentences[-1]["order"]
                    current_para_lines = []
            else:
                current_para_lines.append(line.strip())

        # Don't forget the last paragraph
        if current_para_lines:
            para_text = "".join(current_para_lines).strip()
            if para_text and not is_volume_title(para_text):
                _add_paragraph(
                    para_text, ch_id, all_sentences, ch_para_ranges, global_order
                )
                global_order = all_sentences[-1]["order"]

        chapters.append({
            "id": ch_id,
            "title": ch_title_line,
            "status": "pending",
            "audio_path": "",
            "paragraphs": ch_para_ranges,
        })

    # Step 3.5: Remove false chapters — a real chapter has body content.
    # If a detected "chapter" only has 1 sentence (just the title), it's a false
    # positive from body text like "详见《传国宝玺》第五十章《千钧一发》".
    real_chapters = []
    removed_ids = set()
    for ch in chapters:
        ch_id = ch["id"]
        ch_body_sents = [s for s in all_sentences
                         if s["chapter_id"] == ch_id and s["type"] != "chapter_title"]
        if len(ch_body_sents) == 0:
            # Remove title sentence and mark for deletion
            all_sentences[:] = [s for s in all_sentences if s["chapter_id"] != ch_id]
            removed_ids.add(ch_id)
        else:
            real_chapters.append(ch)
    if removed_ids:
        print(f"   ⚠️  过滤 {len(removed_ids)} 个误判章节（仅标题无正文）: {sorted(removed_ids)}")

    # Step 4: Fix sentence IDs to use correct format
    # Re-ID all sentences with proper chapter_id and per-chapter numbering
    for sent in all_sentences:
        ch_id = sent["chapter_id"]
        # Count position within chapter (already ordered correctly since chapter_counter
        # is sequential)
        # For now, just fix IDs that were placeholder
        if sent["id"].endswith("-000000") and sent["type"] != "chapter_title":
            # Recalculate
            pass

    # Re-number all sentences with correct per-chapter IDs
    chapter_sent_counters = {}
    for sent in all_sentences:
        ch_id = sent["chapter_id"]
        if sent["type"] == "chapter_title":
            sent["id"] = f"{ch_id}-000000"
            chapter_sent_counters[ch_id] = 1  # next content sentence starts at 1
        else:
            count = chapter_sent_counters.get(ch_id, 0)
            sent["id"] = f"{ch_id}-{count:06d}"
            chapter_sent_counters[ch_id] = count + 1

    # Step 5: Build TOC
    toc = build_toc(real_chapters, all_sentences)

    # Step 6: Build novel dict
    novel = {
        "novel_title": book_name,
        "author": "",
        "source_file": filepath,
        "encoding": encoding,
        "voice_model": "single",  # default, can be overridden
        "voice_profile": {
            "ref_file": "",
            "ref_text": "",
            "nfe_step": 32,
            "speed": 1.0,
            "seed": 42,
            "remove_silence": False,
        },
        "toc": toc,
        "sentences": all_sentences,
    }

    # If front matter exists, note it
    if front_matter_sentences_start:
        toc.insert(0, {
            "type": "front_matter",
            "title": "前言/人物表",
            "sentence_range": [front_matter_sentences_start, front_matter_sentences_end],
        })

    return novel, encoding


def _add_paragraph(para_text, ch_id, all_sentences, ch_para_ranges, current_global_order):
    """Add a paragraph's sentences to all_sentences, handling length-based splitting.

    Recursively splits until every chunk is ≤ 500 chars so each TTS generation
    unit stays within Qwen3-TTS's comfortable input range (max_new_tokens=2048).
    """
    sentences = split_sentences(para_text)
    if not sentences:
        return

    _split_and_emit(sentences, ch_id, all_sentences, ch_para_ranges, max_chars=500)


def _split_and_emit(sentences, ch_id, all_sentences, ch_para_ranges, max_chars=500):
    """Recursively split sentences into chunks ≤ max_chars and emit each chunk."""
    total = sum(len(s) for s in sentences)
    if total <= max_chars or len(sentences) == 1:
        # Base case: chunk is small enough, or a single sentence
        # that can't be split further (split_sentences already tried).
        _emit_para_sentences(sentences, ch_id, all_sentences, ch_para_ranges)
        return

    half = find_split_point(sentences, max_chars=max_chars)
    if half is None or half <= 0 or half >= len(sentences):
        # Can't find a valid split — emit as-is
        _emit_para_sentences(sentences, ch_id, all_sentences, ch_para_ranges)
        return

    left = sentences[:half]
    right = sentences[half:]
    if left:
        _split_and_emit(left, ch_id, all_sentences, ch_para_ranges, max_chars)
    if right:
        _split_and_emit(right, ch_id, all_sentences, ch_para_ranges, max_chars)


def _emit_para_sentences(sentences, ch_id, all_sentences, ch_para_ranges):
    """Emit sentences for a paragraph chunk, tracking paragraph range."""
    para_sents = []
    next_order = (all_sentences[-1]["order"] + 1) if all_sentences else 1
    for s in sentences:
        sent = {
            "id": "",  # filled later
            "order": next_order,
            "chapter_id": ch_id,
            "type": "sentence",
            "text": s,
        }
        if not para_sents:
            sent["status"] = "pending"
            sent["audio_path"] = ""
        para_sents.append(sent)
        next_order += 1

    all_sentences.extend(para_sents)
    start_order = para_sents[0]["order"]
    end_order = para_sents[-1]["order"]
    ch_para_ranges.append([start_order, end_order])


def find_split_point(sentences, max_chars=500):
    """Find a sentence index where splitting would keep both halves ≤ max_chars."""
    if not sentences:
        return None

    total = sum(len(s) for s in sentences)
    running = 0
    for i, s in enumerate(sentences):
        running += len(s)
        if running >= total / 2:
            # Check if first half is ≤ max_chars
            first_half = running
            second_half = total - running
            if first_half <= max_chars * 1.1 and second_half <= max_chars * 1.1:
                return i + 1
            return max(i, 1)  # best effort, but never return 0
    return len(sentences) // 2


def build_toc(chapters, all_sentences):
    """Build TOC structure from chapters and sentences.

    chapters is a list of dicts with keys: id, title, status, audio_path, paragraphs
    """
    ch_list = []
    for ch in chapters:
        ch_id = ch["id"]
        # Find sentence range for this chapter
        ch_sentences = [s for s in all_sentences if s["chapter_id"] == ch_id]
        if ch_sentences:
            min_order = min(s["order"] for s in ch_sentences)
            max_order = max(s["order"] for s in ch_sentences)
        else:
            min_order = 0
            max_order = 0

        ch_list.append({
            "id": ch_id,
            "title": ch["title"],
            "status": ch.get("status", "pending"),
            "audio_path": ch.get("audio_path", ""),
            "sentence_range": [min_order, max_order],
            "paragraphs": ch.get("paragraphs", []),
        })

    return [{
        "volume": 1,
        "title": "正文",
        "parts": [{
            "part": 1,
            "title": "",
            "chapters": ch_list,
        }],
    }]


# ── Manifest generation ─────────────────────────────────────────────

def generate_manifest(novel):
    """Generate manifest.txt content from novel dict."""
    toc = novel.get("toc", [])
    sentences = novel.get("sentences", [])

    lines = []
    lines.append(f"# Chapter Manifest — {novel.get('novel_title', 'Unknown')}")
    lines.append("# 编辑操作列后保存，运行 novel-tts apply 应用")
    lines.append("# 操作: K(keep) | M(merge) | S(split) | X(skip)")
    lines.append("# M: 合并到上一章（删除本章标题）")
    lines.append("# S: 需拆分（用 novel-tts view <ID> 标记拆点）")
    lines.append("# X: 标记为 front_matter，不朗读")
    lines.append("")
    lines.append(f"{'OP':6s} {'ID':6s} {'句数':6s} {'标题'}")
    lines.append(f"{'---':6s} {'----':6s} {'----':6s} {'----'}")

    for vol in toc:
        if vol.get("type") == "front_matter":
            continue
        for part in vol.get("parts", []):
            for ch in part.get("chapters", []):
                ch_id = ch["id"]
                sr = ch.get("sentence_range", [0, 0])
                sent_count = sr[1] - sr[0] + 1 if sr[1] >= sr[0] else 0
                title = ch.get("title", "")
                lines.append(f"{'K':6s} {ch_id:6s} {str(sent_count):6s} {title}")

    return "\n".join(lines)
