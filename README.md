# Document QA Agent

面向扫描版 PDF 的智能文档问答 Agent 原型，实现作业要求的完整闭环：

**PDF 类型检测 → OCR / 结构化解析 → 向量检索 RAG → 带引用生成 → 答案自检 / 拒答**

内置两个评测文档：

| 文档 | doc_id | 场景 |
|------|--------|------|
| GB/T 1568-2008《键 技术条件》 | `GBT+1568-2008+键+技术条件` | 国标扫描件（条款 + 表1） |
| agent开发作业样本 | `agent开发作业样本` | 金融财报（表格数值） |

样例 PDF 位于 `doc/`，运行时请复制到 `data/raw/`。

---

## 功能概览

- **PDF 解析**：自动检测扫描件 / 文本层；支持乱码文本层识别与 `--force-ocr` 强制 PaddleOCR
- **结构化抽取**：OCR 路径下识别条款编号（3.1、3.5）与表格（Markdown + 行级块）
- **向量检索**：Chroma + OpenAI 兼容 Embedding（推荐 DashScope `text-embedding-v4`）
- **可靠问答**：Evidence-only 生成；返回页码 / 条款 / 原文引用
- **自检拒答**：引用校验、数值校验、LLM grounding；无证据 / 幻觉 / 不足时拒答
- **多文档**：统一索引，`--doc-id` 限定检索范围
- **评测**：10 题 golden QA（正文 / 表格 / 无答案 / 跨文档）
- **交互**：Typer CLI + Streamlit 演示

---

## 快速开始

### 1. 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[ocr]"   # 扫描 PDF / OCR 路径必需
cp .env.example .env
# 编辑 .env，填入 API Key 与 Base
```

**推荐配置（阿里云 DashScope）**：

```bash
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-xxx
LLM_MODEL=qwen3.7-plus
EMBEDDING_MODEL=text-embedding-v4
```

> 向量索引依赖 Embedding API。未配置 Key 时 `index` 会报错；DeepSeek 等仅 LLM 的场景需另行配置 Embedding 端点。

### 2. 放置 PDF

```bash
cp "doc/GBT+1568-2008+键+技术条件.pdf" data/raw/
cp "doc/agent开发作业样本.pdf" data/raw/
```

也可在 Streamlit 侧边栏上传 PDF。

### 3. 解析与索引

```bash
# 查看文档状态
python3 -m src.cli docs

# 解析：GBT 国标会自动识别乱码文本层并走 OCR
python3 -m src.cli parse --all

# 若希望两个文档都走 PaddleOCR 结构化（条款/表格切分），加 --force-ocr
python3 -m src.cli parse --all --force-ocr

# 构建向量索引（需可用 Embedding API）
python3 -m src.cli index --all --skip-parse
```

首次 OCR 会下载 PaddleOCR 模型，**每页约 2–3 分钟**，属正常现象。

预览解析结果：

```bash
python3 -m src.cli blocks --limit 5
python3 -m src.cli blocks --doc-id "GBT+1568-2008+键+技术条件"
```

### 4. 问答

```bash
# 国标正文
python3 -m src.cli ask "键的抗拉强度最低要求是多少？" \
  --doc-id "GBT+1568-2008+键+技术条件"

# 国标表格
python3 -m src.cli ask "表1中普通平键的键宽 b 尺寸公差 AQL 是多少？" \
  --doc-id "GBT+1568-2008+键+技术条件"

# 无答案（标准排除花键）
python3 -m src.cli ask "花键的技术要求是什么？" \
  --doc-id "GBT+1568-2008+键+技术条件" --json

# 金融样本
python3 -m src.cli ask "2025年6月30日代理买卖证券款的合计金额是多少？" \
  --doc-id "agent开发作业样本"
```

### 5. 自动化评测

```bash
python3 -m src.cli eval              # 两文档各 5 题，分别出报告
python3 -m src.cli eval --suite gbt  # 仅国标
python3 -m src.cli eval --suite agent
python3 -m src.cli eval --suite combined  # 合并 10 题报告
```

### 6. Streamlit 演示

```bash
streamlit run app/streamlit_app.py
```

侧边栏支持：PDF 上传、强制 OCR 解析、统一索引、多文档问答范围选择。

### 7. 无 PDF 时的 Smoke Test

```bash
python3 -m src.cli seed
python3 -m src.cli ask "键的抗拉强度最低要求是多少？"
```

---

## CLI 命令

| 命令 | 说明 |
|------|------|
| `docs` | 列出 `data/raw/` 文档及解析 / 索引状态 |
| `parse [--all] [--force-ocr] [--pdf PATH]` | 解析 PDF，输出 blocks 到 `data/processed/documents/` |
| `index [--all] [--skip-parse] [--force-ocr]` | 构建 Chroma 向量索引 |
| `ask QUESTION [--doc-id ID] [--json]` | 问答 |
| `eval [--suite gbt\|agent\|combined\|all]` | golden QA 评测 |
| `blocks [--doc-id ID] [--limit N]` | 预览解析块 |
| `seed` | 从 fixtures 注入样例索引（无需 PDF） |

---

## 项目结构

```text
DocumentQAAgent/
├── app/streamlit_app.py          # Web 演示
├── src/
│   ├── pdf/                      # 类型检测、PaddleOCR、条款/表格结构化
│   ├── indexing/                 # 分块、Embedding、Chroma 向量索引
│   ├── agent/                    # 路由、检索、生成、Verifier
│   ├── documents/                # 多文档注册表与管理
│   ├── plugins/base.py           # 领域 Parser 扩展协议
│   └── cli.py                    # 命令行入口
├── tests/
│   ├── eval/                     # golden_qa_*.json + run_eval.py
│   └── fixtures/                 # smoke test 样例数据
├── doc/                          # 作业要求、设计/测试/演示说明、样例 PDF
└── data/
    ├── raw/                      # 运行时 PDF（git 忽略 *.pdf，保留 .gitkeep）
    └── processed/                # blocks、chunks、chroma（git 忽略）
```

---

## PDF 解析策略

系统按以下优先级决定是否走 PaddleOCR：

1. **`--force-ocr` 或 `PDF_FORCE_OCR=true`**：强制 OCR + 条款/表格结构化
2. **低文本页占比 ≥ 80%**：判定为扫描件
3. **乱码 / 全角文本层**：GBT 等国标 PDF 虽含文本层但编码异常，会自动命中
4. **否则**：PyMuPDF 直接抽取整页段落（金融样本默认走此路径）

```bash
# 作业推荐：两个文档统一 OCR 结构化
python3 -m src.cli parse --all --force-ocr
```

---

## 配置说明

| 变量 | 说明 | 默认 |
|------|------|------|
| `OPENAI_API_BASE` | OpenAI 兼容 API 地址 | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | API Key | - |
| `LLM_MODEL` | 生成 / Verifier 模型 | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | 向量模型 | `text-embedding-v4` |
| `RETRIEVAL_TOP_K` | 向量召回数 | `8` |
| `RETRIEVAL_FINAL_K` | 最终送入 LLM 的片段数 | `5` |
| `RETRIEVAL_SCORE_THRESHOLD` | 检索拒答阈值 | `0.35` |
| `VERIFIER_MIN_GROUNDING` | 自检最低置信度 | `0.7` |
| `OCR_DPI` | OCR 渲染 DPI | `300` |
| `PDF_FORCE_OCR` | 全局强制 OCR | `false` |
| `PDF_BAD_TEXT_PAGE_THRESHOLD` | 单页乱码阈值 | `0.15` |
| `PDF_BAD_TEXT_AVG_THRESHOLD` | 文档平均乱码阈值 | `0.10` |

完整示例见 [`.env.example`](.env.example)。

---

## 架构简述

```text
Query → QueryRouter（表格/条款/超范围/通用）
      → VectorRetriever（Chroma + Embedding，表格/条款加权）
      → AnswerGenerator（Evidence-only + citations）
      → AnswerVerifier（引用 / 数值 / grounding 三重校验）
      → QAResponse（answered | refused）
```

详细设计见 [`doc/设计说明.md`](doc/设计说明.md)，测试方法见 [`doc/测试说明.md`](doc/测试说明.md)，演示脚本见 [`doc/演示说明.md`](doc/演示说明.md)。

---

## AI 工具使用说明

本项目使用 **Cursor** 辅助开发，各环节均经人工校验：

| 环节 | AI 用途 | 人工校验 |
|------|---------|----------|
| 模块骨架 | 生成目录与接口 | 对照作业要求裁剪 scope |
| Prompt | 迭代生成 / Verifier 提示词 | golden_qa 回归 |
| OCR 检测 | 乱码文本层启发式 | 对照 blocks.json spot-check |
| 评测集 | 初稿测试题 | 按标准正文修正 expected_keywords |

---

## License

MIT
