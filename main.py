"""
Backward-compatible entry point for running the application via `python main.py`.
"""

from chatpdfv1.interfaces.cli import main as cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())
