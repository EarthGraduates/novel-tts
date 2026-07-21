#!/usr/bin/env python3
"""Preprocess 大力金刚掌-茅山后裔.txt: merge two-line chapter titles into one.

Source format (two lines):
  　　第一卷　传国宝玺·第一部　撞客
  　　茅山后裔第01章　《茅山图志》
  OR single line:
  　　茅山后裔第一章 情敌

Target format (one line):
  第一卷 传国宝玺 第一部 撞客 第1章 《茅山图志》
  第1章 情敌

Usage:
  python3 preprocess_novel.py
"""

import re
import sys

INPUT = "/home/phoenix/ClaudeProjects/TTS/大力金刚掌-茅山后裔.txt"
OUTPUT = "/home/phoenix/ClaudeProjects/TTS/大力金刚掌-茅山后裔_clean.txt"


# ── Chinese numeral → int ──

CN_NUM = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000,
}


def cn_to_int(s):
    """Convert Chinese numeral string to integer. e.g. '五十八' → 58, '一百二十三' → 123."""
    s = s.strip()
    # Try Arabic first
    if s.isdigit():
        return int(s)
    total = 0
    seg = 0
    for ch in s:
        if ch not in CN_NUM:
            return None
        n = CN_NUM[ch]
        if n >= 10:
            if seg == 0:
                seg = 1
            total += seg * n
            seg = 0
        else:
            seg = n
    total += seg
    return total if total > 0 else None


# ── Patterns ──

# Volume/part header: 第X卷　书名·第X部　部名
VOL_PAT = re.compile(
    r"^[\s　]*"
    r"第[零一二三四五六七八九十百千\d]+卷"
    r"(?:[\s　]+[^\s　·]+)?"
    r"(?:·第[零一二三四五六七八九十百千\d]+部[\s　]+[^\s　]+)?"
    r"[\s　]*$"
)

# Chapter line: 茅山后裔第XX章　标题  (Arabic + Chinese numerals)
#   Space after chapter marker is optional — some chapters lack it (e.g. 茅山后裔第四十一章逆流而上)
CHAP_PAT = re.compile(
    r"^[\s　]*茅山后裔第([零一二三四五六七八九十百千\d]+)[章回节集][\s　]*(.*)$"
)

# False chapter patterns (body text that looks like a chapter but isn't)
FALSE_CHAP_PAT = re.compile(
    r"茅山后裔第[零一二三四五六七八九十百千\d]+[章回节集][。，！？…]"
)

# Real chapter keywords (序/后记 etc — these ARE chapters, not front matter)
REAL_CHAPTER_KW = ["序", "后记", "前言", "尾声", "楔子", "引子", "终章", "大结局"]

# Pure reference material (NOT chapters)
REF_ONLY_KW = ["人物志", "人物表", "角色介绍", "设定集", "内容简介"]

# End-of-volume: （《茅山后裔》第X卷"XXX"全文完）
VOL_END_PAT = re.compile(r"《茅山后裔》第[零一二三四五六七八九十百千\d]+卷.*全文完")

# Duplicate ebook banner
BANNER_PAT = re.compile(r"老云书库|www\.laoyun\.net|电子书.*下载|txt电子书")

# Body text reference (false positive for vol detection)
FALSE_VOL_PAT = re.compile(r"详见|参见|参考|见《")


def is_vol_line(line):
    s = line.strip()
    if not s or len(s) < 8:
        return False
    if not VOL_PAT.match(s):
        return False
    if FALSE_VOL_PAT.search(s):
        return False
    return True


def is_chap_line(line):
    s = line.strip()
    if not CHAP_PAT.match(s):
        return False
    # Filter body text false positives like "茅山后裔第五十八章。"
    if FALSE_CHAP_PAT.search(s):
        return False
    return True


def is_front_matter_only(title):
    """Check if a chapter title is pure reference material (not a real chapter)."""
    # If it contains a real chapter keyword (序/后记 etc), it's a real chapter
    for kw in REAL_CHAPTER_KW:
        if kw in title:
            return False
    # If it contains reference keywords, it's front matter
    for kw in REF_ONLY_KW:
        if kw in title:
            return True
    # "作品相关" alone is front matter
    if "作品相关" in title:
        return True
    return False


def parse_chap_line(line):
    """Extract (chapter_num_str, chapter_num_int, title) from chapter line."""
    m = CHAP_PAT.match(line.strip())
    if m:
        num_str = m.group(1)
        title = m.group(2).strip()
        num_int = cn_to_int(num_str)
        return num_str, num_int, title
    return None, None, None


def parse_vol_line(line):
    s = line.strip()
    s = s.replace("　", " ").replace("·", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def main():
    with open(INPUT, "r", encoding="gbk") as f:
        raw_lines = f.readlines()

    cleaned_lines = []
    current_vol = ""
    stats = {"vol_lines": 0, "chap_lines": 0, "merged": 0, "banners_removed": 0,
             "vol_end_markers": 0, "fm_chapters": 0, "standalone_chapters": 0}

    for i, line in enumerate(raw_lines):
        stripped = line.strip()

        # Skip ebook banners
        if BANNER_PAT.search(stripped):
            stats["banners_removed"] += 1
            continue

        # Skip end-of-volume markers
        if VOL_END_PAT.search(stripped):
            stats["vol_end_markers"] += 1
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        # Volume/part header — remember for next chapter line
        if is_vol_line(stripped):
            current_vol = parse_vol_line(stripped)
            stats["vol_lines"] += 1
            continue

        # Chapter title line
        num_str, num_int, title = parse_chap_line(stripped)
        if num_str is not None:
            stats["chap_lines"] += 1

            # Front matter (纯参考材料，不朗读)
            if is_front_matter_only(title):
                prefix = "" if title.startswith("作品相关") else "作品相关 "
                cleaned_lines.append(f"\n{prefix}{title}\n")
                stats["fm_chapters"] += 1
                current_vol = ""
                continue

            # Special chapters (序/后记 etc) without volume context — no chapter number needed
            is_special = any(kw in title for kw in REAL_CHAPTER_KW)
            ch_num = num_int if num_int else num_str

            if current_vol:
                cleaned_lines.append(f"\n{current_vol} 第{ch_num}章 {title}\n")
                stats["merged"] += 1
            elif is_special:
                # Strip "作品相关 " so parser's ^序 / ^后记 patterns match
                clean = title
                for prefix in ["作品相关 ", "作品相关"]:
                    if clean.startswith(prefix):
                        clean = clean[len(prefix):]
                cleaned_lines.append(f"\n{clean}\n")
                stats["standalone_chapters"] += 1
            else:
                cleaned_lines.append(f"\n第{ch_num}章 {title}\n")
                stats["standalone_chapters"] += 1
            continue

        # Normal body text
        if stripped:
            cleaned_lines.append(stripped)
        else:
            cleaned_lines.append("")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines))

    print(f"Input:  {len(raw_lines)} lines (GBK)")
    print(f"Output: {OUTPUT} (UTF-8)")
    print()
    print(f"Stats:")
    print(f"  卷/部标题:        {stats['vol_lines']}")
    print(f"  章节标题:         {stats['chap_lines']}")
    print(f"  合并(卷+章):      {stats['merged']}")
    print(f"  独立章节(无卷标): {stats['standalone_chapters']}")
    print(f"  参考材料(不朗读): {stats['fm_chapters']}")
    print(f"  广告删除:         {stats['banners_removed']}")
    print(f"  卷尾标记:         {stats['vol_end_markers']}")


if __name__ == "__main__":
    main()
