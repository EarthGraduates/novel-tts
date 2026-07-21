"""novel-tts status — Read novel.json and print generation progress."""

import os
import sys

from tts_core.config import ensure_novels_dirs, load_novel


def run(args):
    novels_dir = ensure_novels_dirs()

    # Find novel.json
    if args:
        book_name = args[0].replace("_novel.json", "").replace(".json", "")
    else:
        novels = [f for f in os.listdir(novels_dir) if f.endswith("_novel.json")]
        if not novels:
            print("❌ 未找到 *_novel.json，请先运行 novel-tts parse 和 novel-tts generate")
            sys.exit(1)
        if len(novels) > 1:
            for n in novels:
                name = n.replace("_novel.json", "")
                _print_status(name)
            return
        book_name = novels[0].replace("_novel.json", "")

    _print_status(book_name)


def _print_status(book_name):
    novel = load_novel(book_name)
    if not novel:
        print(f"❌ {book_name}: novel.json 不存在")
        return

    sentences = novel.get("sentences", [])
    para_starts = [s for s in sentences if "status" in s]

    total = len(para_starts)
    done = sum(1 for s in para_starts if s["status"] == "done")
    errors = sum(1 for s in para_starts if s["status"] == "error")
    failed = sum(1 for s in para_starts if s["status"] == "failed")
    pending = total - done - errors - failed

    # Count chapter status
    toc = novel.get("toc", [])
    ch_total = 0
    ch_done = 0
    for vol in toc:
        if vol.get("type") == "front_matter":
            continue
        for part in vol.get("parts", []):
            for ch in part.get("chapters", []):
                ch_total += 1
                if ch.get("status") == "done":
                    ch_done += 1

    pct = (done / total * 100) if total > 0 else 0

    print()
    print(f"📖 {novel.get('novel_title', book_name)} — 生成进度")
    print(f"   {'─' * 35}")
    print(f"   段落:   {done}/{total} 完成 ({pct:.0f}%)")
    print(f"   章节:   {ch_done}/{ch_total} 拼接完成")
    if pending:
        print(f"   待处理: {pending} 段")
    if errors:
        print(f"   错误:   {errors} 段 → 运行 novel-tts generate 断点续传")
    if failed:
        print(f"   失败:   {failed} 段 → 需手动处理")
    print()
    if done == total and ch_done == ch_total:
        print("   ✅ 全部完成！")
        # Find output
        output_dir = os.path.join(os.getcwd(), "novels", "output", book_name)
        if os.path.exists(output_dir):
            print(f"   📁 {output_dir}")
