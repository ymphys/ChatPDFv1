"""
Core business logic for the ChatPDFv1 application.
"""

from .interpreter import chatgpt_interpretation  # noqa: F401

__all__ = ["chatgpt_interpretation"]
