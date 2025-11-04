import requests
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import time
import logging

# Logging: brief info to console, detailed debug to file
LOG_FILE = "./output/chatmd.log"
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

def load_existing_answers(path="./output/interpretation_results.md"):
    """
    读取已有的 interpretation_results.md，提取已经存在的问答问题（以避免重复处理）
    返回一个 set，包含已回答的问题文本（尽量保持与问题源文本一致的匹配）
    """
    existing = set()
    if not os.path.exists(path):
        return existing
    try:
        with open(path, 'r', encoding='utf-8') as f:
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

def chatgpt_interpretation(md_content, questions, openai_api_key):
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
    existing = load_existing_answers()

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
                    logger.debug(f"Chunk {i} preview: {text[:120].replace('\n', ' ')}")
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
    output_path = "./output/interpretation_results.md"
    try:
        if new_sections:
            if not os.path.exists(output_path):
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("# 文档解读\n\n")
                    f.write(new_sections)
            else:
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(new_sections)
            logger.info(f"Interpretation answers appended to {output_path}")
        else:
            logger.info("No new interpretation sections to write (all questions were already present).")
    except Exception as e:
        logger.error(f"Error saving interpretation: {e}")

    return new_sections

def main():
    logger.info("Starting Chatmd main process")
    OPENAI_API_KEY = load()
    md_file_path = "./mds/9711200v3_MinerU__20251101031155.md"
    md_content = read_md_content(md_file_path)

    questions = [
        "请用以下模板概括该文档，并将其中的占位符填入具体信息；若文中未提及某项，请写‘未说明’；若涉及到专业词汇，请在结尾处统一进行解释：[xxxx年]，[xx大学/研究机构]的[xx作者等]针对[研究问题]，采用[研究手段/方法]，对[研究对象或范围]进行了研究，并发现/得出[主要结论]。"
        # ,"添加其他问题..."
    ]
    chatgpt_interpretation(md_content, questions, OPENAI_API_KEY)
    logger.info("Chatmd main process finished")

if __name__ == "__main__":
    main()    
