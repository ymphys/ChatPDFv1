import argparse
import json
import logging
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlparse

import requests
from dotenv import load_dotenv

# Logging: brief info to console, detailed debug to file
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "chatmd.log"
logger = logging.getLogger("chatmd")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

def load():
    """
    加载环境变量中的 OpenAI API Key
    """
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError("未找到 OPENAI_API_KEY 环境变量，请确保已设置系统环境变量")
    return OPENAI_API_KEY

def read_md_content(file_path):
    """
    直接读取md文件全文内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return {'content': content}
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None
    
def split_into_chunks(content, chunk_size=100000):
    """
    将内容分成多个块，每块不超过指定字符数（默认100000）
    """
    return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

def _post_with_retries(url, headers, json_data, max_retries=4, base_delay=1):
    """
    发送 POST 请求，遇到 429/5xx 时重试（指数退避），返回 requests.Response
    并记录每次API调用的token用量和价格（如有usage字段）
    """
    # GPT-4 Turbo价格（2025年10月）
    PRICE_INPUT_PER_1K = 0.01  # 美元/千tokens
    PRICE_OUTPUT_PER_1K = 0.03
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=json_data, timeout=30)
            if resp.status_code == 200:
                logger.debug(f"POST {url} success (200)")
                # 记录 token 用量和价格
                try:
                    usage = resp.json().get('usage', {})
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    cost = (prompt_tokens / 1000 * PRICE_INPUT_PER_1K) + (completion_tokens / 1000 * PRICE_OUTPUT_PER_1K)
                    logger.info(f"API用量: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}, 估算价格=${cost:.4f}")
                except Exception as e:
                    logger.warning(f"无法解析API用量: {e}")
                return resp
            # 对于429和常见5xx做重试
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"OpenAI API returned {resp.status_code}, attempt {attempt}/{max_retries}")
                if attempt < max_retries:
                    time.sleep(base_delay * (2 ** (attempt - 1)))
                    continue
            # 非重试场景或最后一次重试，直接返回响应
            logger.debug(f"POST {url} returned status {resp.status_code}")
            return resp
        except requests.RequestException as e:
            logger.error(f"Request exception on attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** (attempt - 1)))
                continue
            raise

def load_existing_answers(path: Path):
    """
    读取已有的 interpretation_results.md，提取已经存在的问答问题（以避免重复处理）
    返回一个 set，包含已回答的问题文本（尽量保持与问题源文本一致的匹配）
    """
    existing = set()
    if not path.exists():
        return existing
    try:
        with path.open('r', encoding='utf-8') as f:
            content = f.read()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith('## '):
                qline = line[3:].strip()
                # 支持两种形式："Q: ..." 或 直接问题文本
                if qline.startswith('Q:'):
                    q = qline[2:].strip()
                else:
                    q = qline
                if q:
                    existing.add(q)
    except Exception as e:
        logger.error(f"Failed to read existing answers from {path}: {e}")
    return existing

def chatgpt_interpretation(md_content, questions, openai_api_key, output_path: Path):
    """
    使用ChatGPT对md内容进行解读
    questions: 固定问题列表
    结果按md格式输出保存为"{md-filename}-interp.md"
    """
    if not md_content:
        logger.info("No content to interpret")
        return
    
    # OpenAI API设置
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    
    # 读取已存在的答案，避免重复处理同一问题
    existing = load_existing_answers(output_path)

    # 收集新生成的部分，最后统一写入/追加到 interpretation_results.md
    new_sections = ""

    for question in questions:
        if question in existing:
            logger.info(f"Skipping interpretation for question (already present): {question}")
            continue
        try:
            # 按块发送，每块单独获取针对该块的回答片段
            chunks = split_into_chunks(md_content['content'])
            partial_answers = []
            for i, chunk in enumerate(chunks, start=1):
                data = {
                    "model": "gpt-4-turbo-preview",
                    "messages": [
                        {"role": "system", "content": "你是一个学术文献分析专家，请基于提供的文档内容回答问题，请注意对专业名词做出解释。"},
                        {"role": "user", "content": f"文档片段 {i}/{len(chunks)}：\n\n{chunk}\n\n问题：{question}"}
                    ],
                    "temperature": 0.7
                }
                resp = _post_with_retries("https://api.openai.com/v1/chat/completions", headers, data)
                if resp is None:
                    partial_answers.append("[请求失败，未获得该片段回答]")
                    logger.warning(f"No response for chunk {i}/{len(chunks)} for question: {question}")
                elif resp.status_code == 200:
                    text = resp.json()['choices'][0]['message']['content'].strip()
                    partial_answers.append(text)
                    logger.info(f"Chunk {i}/{len(chunks)} answered for question: {question}")
                    preview = text[:120].replace("\n", " ")
                    logger.debug(f"Chunk {i} preview: {preview}")
                else:
                    logger.error(f"Error with OpenAI API for chunk {i}: {resp.status_code}")
                    partial_answers.append(f"[片段调用失败：{resp.status_code}]")
                time.sleep(1)  # 保持速率限制

            chunk_count = len(chunks)
            if chunk_count == 1 and partial_answers:
                final_answer = partial_answers[0]
                logger.debug(f"Single chunk for question '{question}', skipping synthesis call.")
                new_sections += f"## {question}\n\n{final_answer}\n\n"
            else:
                # 把各片段回答合并为最终答案（再调用一次让模型整合）
                synth_prompt = (
                    "请基于下面各片段回答，综合出一个简洁、连贯且基于文档的最终回答；若文档未提供信息请明确说明。\n\n"
                    + "\n\n---\n\n".join(partial_answers)
                )
                synth_data = {
                    "model": "gpt-4-turbo-preview",
                    "messages": [
                        {"role": "system", "content": "你负责把分片回答合并成最终答案。"},
                        {"role": "user", "content": f"{synth_prompt}\n\n问题：{question}"}
                    ],
                    "temperature": 0.0
                }
                resp = _post_with_retries("https://api.openai.com/v1/chat/completions", headers, synth_data)
                if resp and resp.status_code == 200:
                    final_answer = resp.json()['choices'][0]['message']['content'].strip()
                    logger.info(f"Synthesized final answer for question: {question}")
                    new_sections += f"## {question}\n\n{final_answer}\n\n"
                else:
                    logger.error(f"Failed to synthesize answer for question '{question}': {resp.status_code if resp else 'no response'}")
                    new_sections += f"## {question}\n\n无法获取答案，API调用失败。\n\n"
        except Exception as e:
            logger.exception(f"Error processing question '{question}': {e}")
            new_sections += f"## {question}\n\n处理此问题时发生错误。\n\n"
    # 将新生成部分追加到文件（若无则创建）
    try:
        if new_sections:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not output_path.exists():
                with output_path.open('w', encoding='utf-8') as f:
                    f.write("# 文档解读\n\n")
                    f.write(new_sections)
            else:
                with output_path.open('a', encoding='utf-8') as f:
                    f.write(new_sections)
            logger.info(f"Interpretation answers appended to {output_path}")
        else:
            logger.info("No new interpretation sections to write (all questions were already present).")
    except Exception as e:
        logger.error(f"Error saving interpretation: {e}")

    return new_sections

def _mineru_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _request_with_retries(method, url, *, max_retries=3, base_delay=2, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.request(method, url, timeout=30, **kwargs)
            if resp.status_code == 200:
                return resp
            logger.warning(
                "MinerU API %s %s returned status %s (attempt %s/%s)",
                method,
                url,
                resp.status_code,
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
    # Prefer the largest markdown file assuming it contains the main content
    markdown_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    selected = markdown_files[0]
    logger.info("Selected markdown file %s from MinerU results", selected)
    return selected


def process_pdf_via_mineru(
    pdf_url: str,
    output_root: Path,
    api_key: str,
    poll_interval: int = 5,
    timeout_seconds: int = 600,
) -> Path:
    base_url = "https://mineru.net/api/v4"
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
    resp = _request_with_retries(
        "POST", f"{base_url}/extract/task", json=payload, headers=headers
    )
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"MinerU task submission failed: {data}")
    task_id = data["data"]["task_id"]
    logger.info("MinerU task created: %s", task_id)

    deadline = time.time() + timeout_seconds
    task_info = None
    while time.time() < deadline:
        resp = _request_with_retries(
            "GET", f"{base_url}/extract/task/{task_id}", headers=headers
        )
        task_data = resp.json()
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

    zip_url = task_info.get("full_zip_url")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Process markdown or remote PDF files.")
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
    return parser.parse_args()


def main():
    logger.info("Starting Chatmd main process")
    args = parse_args()
    OPENAI_API_KEY = load()
    files_root = BASE_DIR / "files"

    if args.pdf_url:
        mineru_api_key = os.getenv("MINERU_API_KEY")
        if not mineru_api_key:
            raise ValueError("MINERU_API_KEY environment variable is not set")
        md_path = process_pdf_via_mineru(
            args.pdf_url,
            output_root=files_root,
            api_key=mineru_api_key,
            timeout_seconds=args.mineru_timeout,
        )
    elif args.md_path:
        md_path = Path(args.md_path)
    else:
        md_path = files_root / "9711200v3_MinerU__20251101031155.md"

    md_content = read_md_content(str(md_path))
    interpretation_output = md_path.parent / "interpretation_results.md"

    questions = [
        "请用以下模板概括该文档，并将其中的占位符填入具体信息；若文中未提及某项，请写‘未说明’；若涉及到专业词汇，请在结尾处统一进行解释：[xxxx年]，[xx大学/研究机构]的[xx作者等]针对[研究问题]，采用[研究手段/方法]，对[研究对象或范围]进行了研究，并发现/得出[主要结论]。"
        # ,"添加其他问题..."
    ]
    chatgpt_interpretation(md_content, questions, OPENAI_API_KEY, interpretation_output)
    logger.info("Chatmd main process finished")


if __name__ == "__main__":
    main()
