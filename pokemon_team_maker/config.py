"""Central URLs and model names."""

from pathlib import Path

# Repository root (parent of pokemon_team_maker package)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/HaoleHawaiian/Pokemon-Team-Maker/main/Data"
)

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
FULL_DEX_DISTILBERT_FILENAME = "full_dex_distilbert.npy"
DISTILBERT_META_FILENAME = "full_dex_distilbert_meta.json"
