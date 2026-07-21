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
    print("║   📖 TTS 有声读物 — 项目初始化   ║")
    print("║      引擎: Qwen3-TTS 0.6B        ║")
    print("╚══════════════════════════════════╝")
    print()

    # ── Book path ──
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

    # ── Speaker selection ──
    print("🔊 选择朗读音色:")
    for i, spk in enumerate(AVAILABLE_SPEAKERS):
        tag = ""
        if spk == "Eric":
            tag = " (四川话)"
        elif spk == "Dylan":
            tag = " (北京话)"
        elif spk == "Uncle_Fu":
            tag = " ← 推荐旁白"
        print(f"   [{i+1}] {spk}{tag}")
    print()

    speaker = DEFAULT_SPEAKER
    while True:
        choice = _ask(f"   选哪个", "6")  # Uncle_Fu is #6
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(AVAILABLE_SPEAKERS):
                speaker = AVAILABLE_SPEAKERS[idx]
                break
        except ValueError:
            if choice in AVAILABLE_SPEAKERS:
                speaker = choice
                break
        print(f"   无效选择，请输入 1-{len(AVAILABLE_SPEAKERS)} 或 speaker 名称")

    # ── Voice instruction ──
    print(f"\n🎭 朗读风格 (instruct):")
    styles = ["narration", "gentle", "sad", "angry", "cheerful", "serious"]
    for i, s in enumerate(styles):
        tag = " ← 推荐旁白" if s == "narration" else ""
        print(f"   [{i+1}] {s}{tag}")
    print(f"   [0] 无（默认音色）")

    instruct = DEFAULT_INSTRUCT
    choice = _ask("   选哪个", "1")
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(styles):
            instruct = styles[idx]
    except ValueError:
        if choice in styles:
            instruct = choice

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

    if not _ask_confirm("   满意吗", "y"):
        print("   请重新运行 novel-tts init 选择其他音色")
        sys.exit(0)

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
