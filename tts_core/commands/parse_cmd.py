"""novel-tts parse — Parse novel text into JSON structure."""

import os
import sys

from tts_core.config import (
    ensure_novels_dirs,
    novel_path,
    manifest_path,
    save_novel,
    load_novel,
    load_config,
    book_name_from_path,
    backup_file,
)
from tts_core.parser import parse, generate_manifest


def run(args):
    # Determine book path
    if args:
        book_path = args[0]
    else:
        # Try current directory for config
        novels_dir = ensure_novels_dirs()
        configs = [f for f in os.listdir(novels_dir) if f.endswith("_config.json")]
        if not configs:
            print("❌ 未找到 *_config.json，请先运行 novel-tts init")
            print("   或: novel-tts parse <小说文件路径>")
            sys.exit(1)
        # Use first config found
        config_file = configs[0]
        config = load_config(config_file.replace("_config.json", ""))
        if not config:
            print(f"❌ 无法加载配置: {config_file}")
            sys.exit(1)
        book_path = config.get("book_path", "")
        if not book_path:
            print("❌ 配置中无 book_path")
            sys.exit(1)

    book_path = os.path.abspath(os.path.expanduser(book_path))
    if not os.path.exists(book_path):
        print(f"❌ 文件不存在: {book_path}")
        sys.exit(1)

    book_name = book_name_from_path(book_path)
    ensure_novels_dirs()

    # Load config if exists
    config = load_config(book_name)

    # Check for existing novel.json
    npath = novel_path(book_name)
    if os.path.exists(npath):
        print(f"⚠️  已有解析结果: {npath}")
        print("   [1] 重新解析（旧文件备份为 .bak）")
        print("   [2] 基于已有 novel.json 继续（跳过解析）")
        choice = input("   选哪个？[1]: ").strip() or "1"
        if choice == "2":
            print(f"✅ 沿用已有解析 → {npath}")
            return
        backup_file(npath)
        mpath = manifest_path(book_name)
        if os.path.exists(mpath):
            backup_file(mpath)
        print("   旧文件已备份")

    # Parse
    print()
    print(f"📖 正在解析: {book_path}")
    print(f"   规则: 包含\"第X章/第X回\" + 前面非》 + 行 ≤ 50字 → 章节标题")
    print(f"   编码检测中...", end=" ", flush=True)

    try:
        novel, encoding = parse(book_path)
    except Exception as e:
        print(f"\n❌ 解析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"✓ ({encoding})")

    ch_count = sum(
        1 for v in novel["toc"]
        if not v.get("type")  # skip front_matter entries
        for p in v.get("parts", [])
        for _ in p.get("chapters", [])
    )
    sent_count = len(novel["sentences"])
    para_count = sum(
        1 for s in novel["sentences"]
        if "status" in s
    )

    print(f"   检测到 {ch_count} 章, {sent_count} 句, {para_count} 段")

    # Merge config voice_profile if available
    if config:
        novel["voice_model"] = config.get("voice_model", "single")
        novel["voice_profile"] = config.get("voice_profile", novel["voice_profile"])

    # Save novel.json
    save_novel(book_name, novel)
    print(f"✅ novel.json → {npath}")

    # Generate manifest
    manifest_content = generate_manifest(novel)
    mpath = manifest_path(book_name)
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(manifest_content)
    print(f"✅ manifest → {mpath}")
    print()
    print(f"   下一步: novel-tts manifest 编辑章节清单")
