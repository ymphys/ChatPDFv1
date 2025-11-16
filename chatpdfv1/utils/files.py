from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("chatpdf")


def read_md_content(file_path: str | Path) -> Optional[Dict[str, str]]:
    """
    Read the markdown file and return its content in a dict structure.
    Returning a dict keeps compatibility with existing consumers.
    """
    path = Path(file_path)
    try:
        content = path.read_text(encoding="utf-8")
        return {"content": content}
    except Exception as exc:
        logger.error("Error reading file %s: %s", path, exc)
        return None


def load_existing_answers(path: Path) -> Dict[str, str]:
    """
    Scan the existing interpretation markdown file and return answered questions.
    The parsing mirrors the single-file implementation to avoid behaviour changes.
    """
    existing: Dict[str, str] = {}
    if not path.exists():
        return existing

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        current_question: Optional[str] = None
        buffer: list[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line.startswith("## "):
                if current_question is not None:
                    buffer.append(raw_line.rstrip())
                continue

            if current_question:
                existing[current_question] = _join_answer(buffer)
                buffer.clear()

            qline = line[3:].strip()
            question = qline[2:].strip() if qline.startswith("Q:") else qline
            current_question = question or None
        if current_question:
            existing[current_question] = _join_answer(buffer)
    except Exception as exc:
        logger.error("Failed to read existing answers from %s: %s", path, exc)
    return existing


def _join_answer(lines: list[str]) -> str:
    """
    Join answer lines while preserving blank lines between paragraphs.
    """
    # Strip trailing empty lines but keep intentional spacing
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines).strip()


__all__ = ["load_existing_answers", "read_md_content"]
