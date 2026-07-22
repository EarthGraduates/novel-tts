"""Config file read/write utilities."""

import json
import os


NOVELS_DIR = "novels"


def ensure_novels_dirs():
    """Ensure the novels/ directory structure exists."""
    base = os.path.join(os.getcwd(), NOVELS_DIR)
    for sub in ["tmp", "speaker"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    return base


def config_path(book_name):
    """Get config file path for a book."""
    return os.path.join(os.getcwd(), NOVELS_DIR, f"{book_name}_config.json")


def novel_path(book_name):
    """Get novel.json path for a book."""
    return os.path.join(os.getcwd(), NOVELS_DIR, f"{book_name}_novel.json")


def manifest_path(book_name):
    """Get manifest.txt path for a book."""
    return os.path.join(os.getcwd(), NOVELS_DIR, f"{book_name}_manifest.txt")


def manifest_detail_path(book_name, chapter_id):
    """Get manifest detail path for a chapter."""
    return os.path.join(os.getcwd(), NOVELS_DIR, f"{book_name}_manifest_{chapter_id}.txt")


def load_config(book_name):
    """Load novel_config.json. Returns dict or None if not found."""
    path = config_path(book_name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(book_name, config):
    """Save novel_config.json."""
    path = config_path(book_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_novel(book_name):
    """Load novel.json. Returns dict or None if not found."""
    path = novel_path(book_name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_novel(book_name, novel):
    """Save novel.json."""
    path = novel_path(book_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(novel, f, ensure_ascii=False, indent=2)


def backup_file(path):
    """Rename path to path.bak if it exists."""
    if os.path.exists(path):
        bak = path + ".bak"
        os.rename(path, bak)
        return bak
    return None


def book_name_from_path(txt_path):
    """Extract book name from txt file path. e.g. '茅山后裔.txt' → '茅山后裔'"""
    basename = os.path.basename(txt_path)
    name, _ = os.path.splitext(basename)
    return name
