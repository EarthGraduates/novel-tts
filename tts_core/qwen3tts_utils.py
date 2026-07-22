"""Backward-compatible re-exports from tts_core.engines.qwen3_engine."""

from tts_core.engines.qwen3_engine import (
    get_model, infer_paragraph, generate_preview, save_preview_wav,
    AVAILABLE_SPEAKERS, DEFAULT_SPEAKER, DEFAULT_INSTRUCT, DEFAULT_LANGUAGE,
)