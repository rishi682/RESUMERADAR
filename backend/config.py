import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def load_project_env() -> bool:
    return load_dotenv(dotenv_path=ENV_PATH, override=False)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def gemini_key_present() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))