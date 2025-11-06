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
    ),
    (
        "本文中建立的GEO磁场模型中，磁场在GEO轨道高度的大概取值范围是多少？请用nT为单位。已知地磁偶极场在GEO轨道高度的磁场强度约为100-300 nT，本文的模型结果与此相比有多大不同？"
    ),
    (
        "本文所建立的磁场模型中，磁场的空间不均匀性如何？比如在GEO高度，百公里范围内是否会有显著变化？磁场随时间的变化情况如何？在秒量级或分钟量级是否会有变化？本文的模型是否会有时效性？比如需要定期更新模型参数？"
    ),
    (
        "本文的磁场模型除了位置坐标外，还需要哪些输入参数，才能够得到磁场的值？这些输入参数应该如何得到？直接在空间进行测量吗？"
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
    logger = logging.getLogger("chatmd")
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
