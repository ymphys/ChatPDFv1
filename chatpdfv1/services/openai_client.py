from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("chatpdf")

PRICE_INPUT_PER_1K = 0.01
PRICE_OUTPUT_PER_1K = 0.03


def post_with_retries(
    url: str,
    headers: Dict[str, str],
    json_data: Dict[str, Any],
    *,
    max_retries: int = 4,
    base_delay: int = 1,
) -> Optional[requests.Response]:
    """
    POST wrapper with retry/backoff tuned for OpenAI API conventions.
    Records usage and estimated pricing when present in the payload.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=json_data, timeout=30)
            if response.status_code == 200:
                logger.debug("POST %s success (200)", url)
                _log_usage(response)
                return response

            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                logger.warning(
                    "OpenAI API returned %s, attempt %s/%s",
                    response.status_code,
                    attempt,
                    max_retries,
                )
                time.sleep(base_delay * (2 ** (attempt - 1)))
                continue

            logger.debug("POST %s returned status %s", url, response.status_code)
            return response
        except requests.RequestException as exc:
            logger.error(
                "Request exception on attempt %s/%s: %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** (attempt - 1)))
                continue
            raise
    return None


def _log_usage(response: requests.Response) -> None:
    """
    Write token usage and cost estimation to the logger if usage metadata exists.
    """
    try:
        usage = response.json().get("usage", {})
    except Exception as exc:
        logger.warning("无法解析API用量: %s", exc)
        return

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    cost = (prompt_tokens / 1000 * PRICE_INPUT_PER_1K) + (
        completion_tokens / 1000 * PRICE_OUTPUT_PER_1K
    )
    logger.info(
        "API用量: prompt_tokens=%s, completion_tokens=%s, total_tokens=%s, 估算价格=$%.4f",
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cost,
    )


__all__ = ["post_with_retries"]
