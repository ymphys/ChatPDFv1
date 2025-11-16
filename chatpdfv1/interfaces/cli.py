from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from ..config import get_settings
from ..core import chatgpt_interpretation
from ..logging import configure_logging
from ..services import process_pdf_via_mineru
from ..utils import read_md_content

QUESTIONS = [
    (
        "请用以下模板概括该文档，并将其中的占位符填入具体信息；若文中未提及某项，请写‘未说明’；"
        "若涉及到专业词汇，请在结尾处统一进行解释：[xxxx年]，[xx大学/研究机构]的[xx作者等]"
        "针对[研究问题]，采用[研究手段/方法]，对[研究对象或范围]进行了研究，并发现/得出[主要结论]。"
    )
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process markdown documents or remote PDF files."
    )
    parser.add_argument(
        "--pdf-url",
        help="URL of the PDF to process with MinerU before analysis.",
    )
    parser.add_argument(
        "--md-path",
        help="Path to an existing markdown file to process directly.",
    )
    parser.add_argument(
        "--mineru-timeout",
        type=int,
        default=600,
        help="Maximum seconds to wait for MinerU extraction to finish.",
    )
    return parser.parse_args(args=argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    logger = logging.getLogger("chatpdf")
    logger.info("Starting ChatPDFv1 CLI process")

    args = parse_args(argv)
    settings = get_settings()
    files_root = settings.files_root

    files_root.mkdir(parents=True, exist_ok=True)

    if args.pdf_url:
        if not settings.mineru_api_key:
            raise ValueError("MINERU_API_KEY environment variable is not set")
        md_path = process_pdf_via_mineru(
            args.pdf_url,
            output_root=files_root,
            api_key=settings.mineru_api_key,
            timeout_seconds=args.mineru_timeout,
        )
    elif args.md_path:
        md_path = Path(args.md_path)
    else:
        md_path = settings.default_md_path

    md_content = read_md_content(md_path)
    interpretation_output = md_path.parent / "interpretation_results.md"

    chatgpt_interpretation(
        md_content,
        QUESTIONS,
        settings.openai_api_key,
        interpretation_output,
    )
    logger.info("ChatPDFv1 CLI process finished")
    return 0


__all__ = ["main", "parse_args"]
