"""
ChatPDFv1 package bootstrap.

Expose high-level helpers so callers can import from `chatpdfv1` without
needing to traverse the entire package hierarchy.
"""

from .config.settings import get_settings, Settings  # noqa: F401

__all__ = ["Settings", "get_settings"]
