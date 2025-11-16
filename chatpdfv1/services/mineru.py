from __future__ import annotations

import logging
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
from urllib.parse import unquote, urlparse

import requests

logger = logging.getLogger("chatpdf")

BASE_URL = "https://mineru.net/api/v4"


def process_pdf_via_mineru(
    pdf_url: str,
    *,
    output_root: Path,
    api_key: str,
    poll_interval: int = 5,
    timeout_seconds: int = 600,
) -> Path:
    """
    Submit a PDF to MinerU, poll until complete, and return the resulting markdown path.
    """
    headers = _mineru_headers(api_key)
    parsed_url = urlparse(pdf_url)
    original_name = Path(unquote(parsed_url.path)).name or "document.pdf"
    stem = _sanitize_basename(Path(original_name).stem)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    task_label = f"{stem}_{timestamp}"
    target_dir = output_root / task_label
    target_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "url": pdf_url,
        "is_ocr": False,
        "enable_formula": True,
        "enable_table": True,
    }
    logger.info("Submitting MinerU extraction task for %s", pdf_url)

    submission = _request_with_retries(
        "POST",
        f"{BASE_URL}/extract/task",
        json=payload,
        headers=headers,
    ).json()
    if submission.get("code") != 0:
        raise RuntimeError(f"MinerU task submission failed: {submission}")

    task_id = submission["data"]["task_id"]
    logger.info("MinerU task created: %s", task_id)

    deadline = time.time() + timeout_seconds
    task_info: Dict[str, str] | None = None
    while time.time() < deadline:
        task_data = _request_with_retries(
            "GET",
            f"{BASE_URL}/extract/task/{task_id}",
            headers=headers,
        ).json()
        if task_data.get("code") != 0:
            raise RuntimeError(f"MinerU task query failed: {task_data}")

        task_info = task_data["data"]
        state = task_info.get("state")
        logger.info("MinerU task %s state: %s", task_id, state)

        if state == "done":
            break
        if state == "failed":
            raise RuntimeError(
                f"MinerU task {task_id} failed: {task_info.get('err_msg', 'unknown reason')}"
            )
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"Timed out waiting for MinerU task {task_id} to finish")

    zip_url = task_info.get("full_zip_url") if task_info else None
    if not zip_url:
        raise RuntimeError("MinerU task completed but no result package URL provided")

    with TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "result.zip"
        _download_file(zip_url, zip_path)
        markdown_path = _extract_markdown_from_zip(zip_path, target_dir)

    pdf_destination = target_dir / f"{task_label}.pdf"
    try:
        _download_file(pdf_url, pdf_destination)
    except Exception as exc:
        logger.warning("Failed to download original PDF %s: %s", pdf_url, exc)

    logger.info(
        "MinerU processing complete. Markdown: %s, PDF: %s",
        markdown_path,
        pdf_destination,
    )
    return markdown_path


def _mineru_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _request_with_retries(
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    base_delay: int = 2,
    **kwargs,
) -> requests.Response:
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            if response.status_code == 200:
                return response
            logger.warning(
                "MinerU API %s %s returned status %s (attempt %s/%s)",
                method,
                url,
                response.status_code,
                attempt,
                max_retries,
            )
        except requests.RequestException as exc:
            logger.warning(
                "MinerU API %s %s request error on attempt %s/%s: %s",
                method,
                url,
                attempt,
                max_retries,
                exc,
            )
        if attempt < max_retries:
            time.sleep(base_delay * (2 ** (attempt - 1)))
    raise RuntimeError(f"MinerU API request failed after {max_retries} attempts: {url}")


def _sanitize_basename(name: str) -> str:
    stem = re.sub(r"[^\w.\-]+", "_", name).strip("._")
    return stem or "document"


def _download_file(url: str, destination: Path) -> None:
    logger.info("Downloading file from %s to %s", url, destination)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)


def _extract_markdown_from_zip(zip_path: Path, target_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    markdown_files = sorted(target_dir.rglob("*.md"))
    if not markdown_files:
        raise FileNotFoundError("No markdown file found in MinerU result package")
    markdown_files.sort(key=lambda item: item.stat().st_size, reverse=True)
    selected = markdown_files[0]
    logger.info("Selected markdown file %s from MinerU results", selected)
    return selected


__all__ = ["process_pdf_via_mineru"]
