from __future__ import annotations


def split_into_chunks(content: str, *, chunk_size: int = 100_000) -> list[str]:
    """
    Split content into uniform chunks. Defaults mirror legacy behaviour.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]


__all__ = ["split_into_chunks"]
