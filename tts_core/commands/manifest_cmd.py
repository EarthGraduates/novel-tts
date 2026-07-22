"""novel-tts manifest — Open chapter manifest for editing."""

import os
import sys
import subprocess

from tts_core.config import (
    ensure_novels_dirs,
    manifest_path,
    load_novel,
)


def get_editor():
    """Get the user's preferred editor."""
    return os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))


def run(args):
    novels_dir = ensure_novels_dirs()

    # Find the novel/manifest
    if args:
        book_name = args[0].replace(".txt", "").replace("_novel.json", "").replace("_config.json", "")
    else:
        # Try to find novel.json in novels/
        novels = [f for f in os.listdir(novels_dir) if f.endswith("_novel.json")]
        if not novels:
            print("❌ 未找到 *_novel.json，请先运行 novel-tts parse")
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

    mpath = manifest_path(book_name)
    if not os.path.exists(mpath):
        print(f"❌ manifest 不存在: {mpath}")
        print("   请先运行 novel-tts parse")
        sys.exit(1)

    editor = get_editor()
    print(f"📝 正在打开编辑器: {editor}")
    print(f"   文件: {mpath}")
    print()
    print("   操作说明:")
    print("   K  — 保留本章 (keep)")
    print("   M  — 合并到上一章（删除本章标题）")
    print("   S  — 需拆分（用 novel-tts view <ID> 标记拆点）")
    print("   X  — 标记为 front_matter，不朗读")
    print()
    print("   编辑完后运行 novel-tts apply 应用修改")

    subprocess.run([editor, mpath])
