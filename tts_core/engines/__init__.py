"""TTS engine registry and unified interface.

Each engine module exposes:
    name: str          — unique identifier, stored in config
    label: str         — human-readable display name
    needs_gpu: bool    — whether GPU is required
    needs_load: bool   — whether model loading is needed before use

    SPEAKERS: list[dict] — [{id, name, gender, default}]
    STYLES: list[dict]   — [{id, name, default}]

    def load():                  — load model (no-op for cloud engines)
    def infer(text, speaker, style, lang) -> wav, sr:  — generate audio
    def preview(text, speaker, style, lang) -> wav, sr: — short preview
"""

from . import qwen3_engine, edge_engine

ENGINES = {
    qwen3_engine.name: qwen3_engine,
    edge_engine.name: edge_engine,
}

ENGINE_LIST = [qwen3_engine, edge_engine]


def get_engine(name):
    """Get engine module by name. Returns None if not found."""
    return ENGINES.get(name)