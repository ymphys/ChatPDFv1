from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FILES_DIR = BASE_DIR.parent / "files"
DEFAULT_MD_FILENAME = "9711200v3_MinerU__20251101031155.md"


@dataclass(frozen=True)
class Settings:
    """Application configuration bundled in a single object."""

    openai_api_key: str
    mineru_api_key: Optional[str]
    files_root: Path
    default_md_filename: str = DEFAULT_MD_FILENAME

    @property
    def default_md_path(self) -> Path:
        return self.files_root / self.default_md_filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load and cache configuration from environment variables.

    Raises:
        ValueError: if a required key (OPENAI_API_KEY) is missing.
    """
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "未找到 OPENAI_API_KEY 环境变量，请确保已设置系统环境变量"
        )

    mineru_api_key = os.getenv("MINERU_API_KEY")
    files_root = Path(os.getenv("CHATPDF_FILES_ROOT", DEFAULT_FILES_DIR)).expanduser()

    return Settings(
        openai_api_key=openai_api_key,
        mineru_api_key=mineru_api_key,
        files_root=files_root,
    )


__all__ = ["Settings", "get_settings"]
