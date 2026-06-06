# Document QA Agent

面向扫描版 PDF 的智能文档问答 Agent 原型，围绕 **GB/T 1568-2008《键 技术条件》** 实现：

`PDF 类型检测 → OCR/表格结构化 → 混合检索 RAG → 带引用生成 → 答案自检/拒答`

## 功能概览

- 自动判断 PDF 为扫描件或文本层，并选择 OCR / 直接抽取
- PaddleOCR 识别中文正文、条款编号与表格
- Chroma + text-embedding-v4 向量检索
- OpenAI 兼容 API 生成答案，返回页码/片段引用
- Verifier 自检 grounding、引用与数值，低置信度拒答
- CLI 索引/问答/评测 + Streamlit 演示界面

## 快速开始

### 1. 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# OCR 支持（扫描 PDF 必需）
pip install -e ".[ocr]"
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 与 API Base
```

### 2. 放置 PDF

将 PDF 放入 `data/raw/`（支持多个文档，也可在 Streamlit 中上传）：

```text
data/raw/GBT+1568-2008+键+技术条件.pdf
data/raw/agent开发作业样本.pdf
```

### 3. 解析与索引

```bash
python3 -m src.cli docs
python3 -m src.cli parse --all
# 金融样本文本层正常；若需两个文档都走 OCR 结构化解析，加 --force-ocr
python3 -m src.cli parse --all --force-ocr
python3 -m src.cli index --all --skip-parse
```

首次 OCR 会下载 PaddleOCR 模型，耗时取决于机器性能。

### 4. 问答

```bash
python3 -m src.cli ask "键的抗拉强度最低要求是多少？"
python3 -m src.cli ask "花键的技术要求是什么？" --json
```

### 5. 自动化评测

```bash
python3 -m src.cli eval              # 两个文档各 5 题，分别出报告
python3 -m src.cli eval --suite gbt  # 仅国标 5 题
python3 -m src.cli eval --suite agent
```

### 6. Streamlit 演示

```bash
streamlit run app/streamlit_app.py
```

## 项目结构

```text
DocumentQAAgent/
├── app/streamlit_app.py      # Web 演示
├── src/
│   ├── pdf/                  # 类型检测、OCR、条款/表格解析
│   ├── indexing/             # 分块、Embedding、Chroma 向量检索
│   ├── agent/                # 路由、检索、生成、Verifier
│   └── cli.py                # 命令行入口
├── tests/eval/               # golden_qa.json + run_eval.py
├── doc/                      # 作业要求、设计说明、测试说明
└── data/
    ├── raw/                  # 原始 PDF（可放多个，也可 Streamlit 上传）
    └── processed/
        ├── registry.json     # 文档注册表
        └── documents/        # 每个 doc_id 独立 blocks.json
```

## 配置说明

| 变量 | 说明 | 默认 |
|------|------|------|
| `OPENAI_API_BASE` | OpenAI 兼容 API 地址 | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | API Key | - |
| `LLM_MODEL` | 生成/Verifier 模型 | `gpt-4o-mini`（DashScope 可用 `qwen3.7-plus`） |
| `EMBEDDING_MODEL` | 向量模型 | `text-embedding-v4`（DashScope） |
| `RETRIEVAL_SCORE_THRESHOLD` | 检索拒答阈值 | `0.35` |
| `VERIFIER_MIN_GROUNDING` | 自检最低置信度 | `0.7` |

## AI 工具使用说明

本项目使用 **Cursor** 辅助完成：

| 环节 | AI 用途 | 人工校验 |
|------|---------|----------|
| 模块骨架 | 生成目录结构与接口 | 对照作业要求裁剪 scope |
| Prompt | 迭代生成/Verifier 提示词 | golden_qa 回归验证 |
| OCR 后处理 | 建议纠错词典 | 对照 blocks.json 人工 spot-check |
| 评测集 | 初稿 8 题 | 按标准正文修正 expected_keywords |

**修正样例**：初版拒答阈值过高导致 Q8（错别字「拉面强度」）误拒 → 降低 `RETRIEVAL_SCORE_THRESHOLD` 并增强 BM25 模糊匹配。

## 已知限制

- OCR 对低质量扫描页、复杂合并单元格表格可能不完整
- 表格解析采用 OCR bbox 聚类，未使用 MinerU 等重型方案
- Verifier 依赖 LLM，离线环境需配置可用 API
- 演示视频需用户本地录制（解析 + 5 问 + eval 报告）

## 文档

- [设计说明](doc/设计说明.md)
- [测试说明](doc/测试说明.md)
- [演示说明](doc/演示说明.md)
- [作业要求](doc/作业要求.md)

## License

MIT
