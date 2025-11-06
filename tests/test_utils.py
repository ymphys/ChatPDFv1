import sys
from pathlib import Path

# Ensure the project root is importable when running the file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatpdfv1.utils import load_existing_answers, split_into_chunks


def test_split_into_chunks_roundtrip():
    content = "abc" * 10
    chunks = split_into_chunks(content, chunk_size=5)
    assert chunks == ["abcab", "cabca", "bcabc", "abc"]
    assert "".join(chunks) == content


def test_load_existing_answers(tmp_path: Path):
    sample = tmp_path / "interpretation_results.md"
    sample.write_text(
        "# 文档解读\n\n## Q: 问题一\n答案一\n\n## 问题二\n答案二\n",
        encoding="utf-8",
    )

    existing = load_existing_answers(sample)
    assert existing["问题一"] == "答案一"
    assert existing["问题二"] == "答案二"


def test_load_existing_answers_preserves_paragraphs(tmp_path: Path):
    sample = tmp_path / "interpretation_results.md"
    sample.write_text(
        "# 文档解读\n\n## 问题三\n第一段\n\n第二段\n\n",
        encoding="utf-8",
    )

    existing = load_existing_answers(sample)
    assert existing["问题三"] == "第一段\n\n第二段"
