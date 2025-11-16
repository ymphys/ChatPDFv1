# ChatPDFv1 – MinerU-Powered PDF Intelligence

ChatPDFv1是一个命令行工作流，它可以将任何公开可下载的PDF链接或本地PDF文件利用[MinerU](https://mineru.net/)平台转换成Markdown格式文件，并进一步进行用户自定义的互动式快速解读，是对[Chatmd](https://github.com/ymphys/Chatmd)的升级。输入url，它会:

- 将链接通过 MinerU API 上传，处理完成后得到 Markdown 版本 (以及原始的 PDF 文件) 并下载至 `files/<slug>/`
- 运行一个支持用户自定义问题的OpenAIk解读，每次问答结果都将存储下来作为后续提问的输入，因此具有上下文记忆功能
- 保存了可后续追溯的log文件，方便用户查看每次运行操作细节

使用它可以加速文献阅读，生成摘要或提取文献关键信息。

---

## 安装指南

1. **克隆仓库**
   ```bash
   git clone https://github.com/your-account/chatpdfv1.git
   cd chatpdfv1
   ```

2. **创建虚拟环境 (可选但推荐)**
   ```bash
   uv sync
   ```
如果没有安装uv，请参见[uv installation](https://docs.astral.sh/uv/getting-started/installation/)

3. **配置环境变量**
   - `OPENAI_API_KEY` – 用于解读文档.
   - `MINERU_API_KEY` – 用于上传PDF链接到 MinerU , 如果你已经将PDF文件转换为markdown文件，则不需要获取该API key.
   - 上述API_KEY默认直接存储在环境变量中

---

## 使用指南

Mineru API 获取：请至[官方页面](https://mineru.net/apiManage/token)申请。

pdf-url模式
```bash
uv run main.py --pdf-url https://example.com/paper.pdf
```
本地mardown模式
```bash
uv run main.py --md-path files/example.md
```

关键命令行参数:

| 参数 | 描述 |
| --- | --- |
| `--pdf-url URL` | 获取，转换，并分析一个在线 PDF 文件。 |
| `--md-path PATH` | 跳过 MinerU 转换过程，分析已有 Markdown 文件。 |

问题自定义：

更改[cli.py](chatpdfv1/interfaces/cli.py)中的QUESTIONS列表。

运行输出:

- 转换后内容存储在 `files/<slug_timestamp>/`
  - `<slug_timestamp>.pdf` – 原始PDF文件下载
  - 提取出的 Markdown (选择 MinerU 返回的最大的 `.md` 文件)
  - `interpretation_results.md` – 由 OpenAI 生成的 问题/答案 报告
- 处理日志写入到了 `chatpdf.log`中。

---

## 使用的技术

- **Python 3.12+**
- **MinerU API** – 远程 PDF-to-Markdown 转换
- **OpenAI API** – 切块的 问题/答案 合成

---

## 徽章

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![MinerU](https://img.shields.io/badge/MinerU-API-green)](https://mineru.net/)
[![Status](https://img.shields.io/badge/status-active-success)](#project-statusroadmap)

---

## 项目状态 / 开发路线图

- **目前状态:** 持续更新中，通过命令行使用, 单一文件，自定义问题，可追加问题。
- **未来计划开发:**
  - 更丰富的文件选项 (批量模式，如支持)
  - 前端图形界面开发
  - 通过 Docker 打包处理依赖

欢迎您通过 issues 或 pull requests 做出贡献！
