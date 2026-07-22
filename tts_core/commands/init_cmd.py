"""novel-tts init — Interactive project initialization (Qwen3-TTS)."""

import os
import sys

from tts_core.config import (
    ensure_novels_dirs, config_path, save_config, load_config, book_name_from_path,
)
from tts_core.qwen3tts_utils import (
    get_model, generate_preview, save_preview_wav,
    AVAILABLE_SPEAKERS, DEFAULT_SPEAKER, DEFAULT_INSTRUCT, DEFAULT_LANGUAGE,
)

PREVIEW_TEXT = "你好，欢迎使用语音合成。"


def _ask(prompt, default=None):
    d = f" [默认 {default}]" if default is not None else ""
    a = input(f"{prompt}{d}: ").strip()
    return a if a else str(default) if default is not None else ""


def _ask_confirm(prompt, default="y"):
    s = {"y": "[Y/n]", "n": "[y/N]"}.get(default, "[y/n]")
    a = input(f"{prompt} {s}: ").strip().lower()
    return (a or default) == "y"


def run(args):
    novels_dir = ensure_novels_dirs()
    novels_abs = os.path.join(os.getcwd(), "novels")

    print()
    print("╔══════════════════════════════════╗")
    print("║   📖 novel-tts — 项目初始化      ║")
    print("║      引擎: Qwen3-TTS 0.6B        ║")
    print("╚══════════════════════════════════╝")
    print()

    # ── Book path (remember previous) ──
    # Scan existing configs to find previously used book paths
    existing_configs = {}
    for f in os.listdir(novels_dir):
        if f.endswith("_config.json"):
            cfg = load_config(f.replace("_config.json", ""))
            if cfg and cfg.get("book_path"):
                name = f.replace("_config.json", "")
                existing_configs[name] = cfg["book_path"]

    book_path = None
    if existing_configs:
        print("📖 已配置的小说:")
        for i, (name, path) in enumerate(existing_configs.items(), 1):
            exists = "✓" if os.path.exists(path) else "✗ 文件丢失"
            print(f"   [{i}] {name}  → {path}  ({exists})")
        print(f"   [n] 选择新文件")
        print()
        choice = _ask("   使用哪个", "1")
        try:
            idx = int(choice) - 1
            names = list(existing_configs.keys())
            if 0 <= idx < len(names):
                book_path = existing_configs[names[idx]]
                print(f"   使用: {book_path}\n")
        except ValueError:
            pass  # treat as "new file"

    if not book_path:
        # Scan data/ directory for .txt files
        data_dir = os.path.join(os.getcwd(), "data")
        txt_files = []
        if os.path.isdir(data_dir):
            txt_files = sorted([
                f for f in os.listdir(data_dir)
                if f.lower().endswith(".txt")
            ])

        if txt_files:
            print("📖 发现以下小说文件:")
            for i, f in enumerate(txt_files, 1):
                size_mb = os.path.getsize(os.path.join(data_dir, f)) / (1024 * 1024)
                print(f"   [{i}] data/{f}  ({size_mb:.1f} MB)")
            print(f"   [0] 手动输入路径")
            print()
            choice = input("   选哪个？[1]: ").strip() or "1"
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(txt_files):
                    book_path = os.path.join(data_dir, txt_files[idx])
                    print(f"   使用: {book_path}\n")
            except ValueError:
                pass  # fall through to manual input

        if not book_path:
            while True:
                book_path = input("📖 小说文件路径: ").strip()
                if not book_path:
                    print("   请输入文件路径")
                    continue
                book_path = os.path.abspath(os.path.expanduser(book_path))
                if not os.path.exists(book_path):
                    print(f"   文件不存在: {book_path}")
                    continue
                if not book_path.lower().endswith(".txt"):
                    print("   只支持 .txt")
                    continue
                break

    book_name = book_name_from_path(book_path)
    print(f"   书名: {book_name}\n")

    # ── Existing config ──
    existing = load_config(book_name)
    if existing:
        print(f"⚠️  已有配置: {config_path(book_name)}")
        if _ask("   [1] 重新配置  [2] 沿用已有", "2") == "2":
            print(f"\n✅ 沿用已有 → {config_path(book_name)}")
            print(f"   下一步: novel-tts parse {book_path}")
            return
        print("   将覆盖已有配置...\n")

    # ── Load model ──
    print("⏳ 正在加载模型...", flush=True)
    try:
        get_model()
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        sys.exit(1)
    print()

    # ── Speaker + Style + Preview loop ──
    # Gender mapping for speakers
    SPEAKER_GENDER = {
        "Vivian":   "女", "Serena": "女", "Ono_Anna": "女", "Sohee": "女",
        "Uncle_Fu": "男", "Ryan": "男", "Aiden": "男",
        "Eric": "男 · 四川话", "Dylan": "男 · 北京话",
    }
    # Style translations
    STYLE_TRANS = {
        "narration": "旁白", "gentle": "温柔", "sad": "悲伤",
        "angry": "愤怒", "cheerful": "欢快", "serious": "严肃",
    }

    speaker = DEFAULT_SPEAKER
    instruct = DEFAULT_INSTRUCT

    while True:
        # ── Speaker selection ──
        print("🔊 选择朗读音色:")
        for i, spk in enumerate(AVAILABLE_SPEAKERS):
            gender = SPEAKER_GENDER.get(spk, "")
            tag = f"({gender})" if gender else ""
            hint = " ← 推荐旁白" if spk == "Uncle_Fu" else ""
            print(f"   [{i+1}] {spk} {tag}{hint}")
        print()

        choice = _ask("   选哪个", "6")  # Uncle_Fu
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(AVAILABLE_SPEAKERS):
                speaker = AVAILABLE_SPEAKERS[idx]
        except ValueError:
            if choice in AVAILABLE_SPEAKERS:
                speaker = choice
            elif choice:
                print(f"   无效选择，请输入 1-{len(AVAILABLE_SPEAKERS)} 或 speaker 名称")
                continue

        # ── Voice instruction ──
        print(f"\n🎭 朗读风格:")
        styles = list(STYLE_TRANS.keys())
        for i, s in enumerate(styles):
            cn = STYLE_TRANS.get(s, "")
            tag = " ← 推荐旁白" if s == "narration" else ""
            print(f"   [{i+1}] {s}（{cn}）{tag}")
        print(f"   [0] 无（默认音色）")

        choice = _ask("   选哪个", "1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(styles):
                instruct = styles[idx]
        except ValueError:
            if choice in styles:
                instruct = choice
            elif choice not in ("", "0"):
                print(f"   无效选择")
                continue

        # ── Preview ──
        print(f"\n🔊 试听: {speaker} / {instruct}")
        print("   生成中...", end=" ", flush=True)
        try:
            audio, sr = generate_preview(PREVIEW_TEXT, speaker=speaker, instruct=instruct)
            preview_path = os.path.join(novels_abs, "tmp", "preview_qwen3.wav")
            save_preview_wav(audio, preview_path, sr)
            print("✓")
            print(f"   试听: {preview_path}")
        except Exception as e:
            print(f"✗ {e}")
            sys.exit(1)

        print()
        print("   [1] 满意，保存配置")
        print("   [2] 重新选择音色/风格")
        print("   [0] 退出")
        choice = _ask("   选哪个", "1")
        if choice == "1":
            break
        elif choice == "0":
            print("   已取消")
            sys.exit(0)
        # else: loop back to re-select

    # ── Save config ──
    config = {
        "book_path": book_path,
        "voice_model": "single",
        "voice_profile": {
            "engine": "qwen3-tts",
            "speaker": speaker,
            "instruct": instruct,
            "language": DEFAULT_LANGUAGE,
        },
    }
    save_config(book_name, config)
    print(f"\n✅ 配置已保存 → {config_path(book_name)}")
    print(f"   下一步: novel-tts parse {book_path}")
