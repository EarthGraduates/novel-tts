"""tts view <ID> — Open chapter detail for SPLIT marking."""

import os
import sys
import subprocess

from tts_core.config import (
    ensure_novels_dirs,
    manifest_detail_path,
    load_novel,
)


def get_editor():
    return os.environ.get("EDITOR", os.environ.get("VISUAL", "vim"))


def generate_detail(chapter_id, novel):
    """Generate chapter detail file content."""
    sentences = novel.get("sentences", [])

    # Find sentences for this chapter
    ch_sentences = [
        s for s in sentences
        if s["chapter_id"] == chapter_id
    ]
    ch_sentences.sort(key=lambda s: s["order"])

    # Find chapter title from TOC
    ch_title = chapter_id
    for vol in novel.get("toc", []):
        if vol.get("type") == "front_matter":
            continue
        for part in vol.get("parts", []):
            for ch in part.get("chapters", []):
                if ch["id"] == chapter_id:
                    ch_title = ch.get("title", chapter_id)

    para_count = sum(1 for s in ch_sentences if "status" in s)

    lines = []
    lines.append(f"# Chapter {chapter_id} Detail — {ch_title}")
    lines.append(f"# 共 {len(ch_sentences)} 句，{para_count} 段")
    lines.append("# 在要拆分处插入 >>> SPLIT <<< ，保存后运行 tts apply")
    lines.append("# 不要修改文字内容")
    lines.append("")

    for s in ch_sentences:
        if s["type"] == "chapter_title":
            lines.append(f"# === 章标题 ===")
        elif "status" in s:
            lines.append(f"# === 段首 === (order={s['order']})")

        # Word-wrap long text at ~80 chars for readability
        text = s["text"]
        if len(text) > 80:
            # Simple wrap: break at sentence boundaries within text
            wrapped = []
            remaining = text
            while len(remaining) > 80:
                # Find a break point (comma, period) near 80
                brk = 80
                for punct in "，。！？；：、":
                    idx = remaining.rfind(punct, 0, 80)
                    if idx > 40:
                        brk = idx + 1
                        break
                wrapped.append(remaining[:brk])
                remaining = remaining[brk:]
            wrapped.append(remaining)
            for w in wrapped:
                lines.append(f"  {w}")
            lines.append("")
        else:
            lines.append(f"  {text}")
            lines.append("")

    return "\n".join(lines)


def run(args):
    if not args:
        print("用法: tts view <章节ID>")
        print("示例: tts view 0003")
        sys.exit(1)

    chapter_id = args[0].zfill(4)  # normalize to 4-digit

    # Find novel.json
    novels_dir = ensure_novels_dirs()
    novels = [f for f in os.listdir(novels_dir) if f.endswith("_novel.json")]
    if not novels:
        print("❌ 未找到 *_novel.json，请先运行 tts parse")
        sys.exit(1)

    book_name = novels[0].replace("_novel.json", "")
    novel = load_novel(book_name)
    if not novel:
        print(f"❌ 无法加载 novel.json")
        sys.exit(1)

    # Check chapter exists
    ch_found = False
    for vol in novel.get("toc", []):
        if vol.get("type") == "front_matter":
            continue
        for part in vol.get("parts", []):
            for ch in part.get("chapters", []):
                if ch["id"] == chapter_id:
                    ch_found = True
                    break

    if not ch_found:
        print(f"❌ 章节 {chapter_id} 不存在")
        sys.exit(1)

    # Generate detail file
    detail_path = manifest_detail_path(book_name, chapter_id)
    content = generate_detail(chapter_id, novel)

    with open(detail_path, "w", encoding="utf-8") as f:
        f.write(content)

    editor = get_editor()
    print(f"📝 章节 {chapter_id} 详情 → {detail_path}")
    print(f"   编辑器: {editor}")
    print(f"   在要拆分处插入 >>> SPLIT <<<，保存后运行 tts apply")

    subprocess.run([editor, detail_path])
