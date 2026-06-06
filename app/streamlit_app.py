from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.agent.orchestrator import DocumentQAAgent
from src.config.settings import get_settings
from src.documents.manager import DocumentManager
from src.documents.registry import DocumentRegistry


st.set_page_config(page_title="Document QA Agent", layout="wide")
st.title("智能文档问答 Agent")
st.caption("多文档 PDF → OCR 解析 → 混合检索 RAG → 引用与自检")


@st.cache_resource
def get_agent():
    return DocumentQAAgent()


def clear_agent_cache():
    get_agent.clear()


settings = get_settings()
settings.raw_dir.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)
manager = DocumentManager()
registry = DocumentRegistry()

with st.sidebar:
    st.header("文档库")

    uploaded = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("保存上传文件", use_container_width=True):
        for file in uploaded:
            target = settings.raw_dir / file.name
            target.write_bytes(file.getbuffer())
        st.success(f"已保存 {len(uploaded)} 个文件到 {settings.raw_dir}")
        st.rerun()

    doc_rows = manager.list_status()
    if doc_rows:
        st.markdown("**已有文档**")
        for row in doc_rows:
            parsed = "✅" if row["parsed"] else "⏳"
            indexed = "✅" if row["indexed"] else "⏳"
            st.write(f"{parsed}解析 {indexed}索引 | `{row['doc_id']}` | {row['filename']}")
    else:
        st.info(f"请将 PDF 放入 `{settings.raw_dir}` 或使用上方上传")

    st.divider()
    st.subheader("解析与索引")
    force_ocr = st.checkbox(
        "强制 PaddleOCR（跳过文本层直抽）",
        value=settings.pdf_force_ocr,
        help="国标扫描件会自动检测；金融样本文本层正常，需勾选此项或设置 PDF_FORCE_OCR=true",
    )
    if st.button("解析全部 PDF", use_container_width=True):
        with st.spinner("正在解析全部文档..."):
            results = manager.parse_all_raw(force_ocr=force_ocr or None)
        st.success(f"解析完成：{len(results)} 个文档")
        st.rerun()

    if st.button("构建统一索引", use_container_width=True):
        with st.spinner("正在构建 text-embedding-v4 向量索引..."):
            chunk_count = manager.index_documents()
        clear_agent_cache()
        st.success(f"索引完成：{chunk_count} chunks")
        st.rerun()

    indexed_docs = [row for row in doc_rows if row["indexed"]]
    doc_options = {f"{row['filename']} ({row['doc_id']})": row["doc_id"] for row in indexed_docs}
    selected_labels = st.multiselect(
        "问答范围",
        options=list(doc_options.keys()),
        default=list(doc_options.keys()),
        help="默认检索全部已索引文档；可缩小到单个文档",
    )
    active_doc_ids = [doc_options[label] for label in selected_labels]

    st.divider()
    st.markdown("**索引状态**")
    st.write("registry.json", "✅" if registry.registry_path.exists() else "❌")
    st.write("chunks.json", "✅" if settings.chunks_path.exists() else "❌")
    st.write("chroma", "✅" if settings.chroma_dir.exists() else "❌")

tab_qa, tab_parse = st.tabs(["问答", "解析预览"])

with tab_parse:
    records = registry.load_records()
    if records:
        preview_doc_id = st.selectbox(
            "预览文档",
            options=[record.doc_id for record in records],
            format_func=lambda doc_id: next(r.filename for r in records if r.doc_id == doc_id),
        )
        profile, blocks = registry.load_document_blocks(preview_doc_id)
        st.subheader("PDF Profile")
        st.json(profile)
        st.subheader("Blocks Preview")
        block_type = st.selectbox("筛选 block 类型", ["all", "clause", "table", "paragraph"], key="preview_type")
        filtered = blocks if block_type == "all" else [b for b in blocks if b.block_type == block_type]
        for block in filtered[:20]:
            with st.expander(f"{block.block_id} | page={block.page} | type={block.block_type}"):
                st.markdown(block.content)
    else:
        st.info("尚未解析 PDF，请在侧边栏点击「解析全部 PDF」")

with tab_qa:
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    sample_questions = [
        "键的抗拉强度最低要求是多少？",
        "键表面不允许有哪些缺陷？",
        "普通平键的键宽 b 尺寸公差 AQL 是多少？",
        "花键的技术要求是什么？",
        "该标准规定的螺栓拧紧力矩是多少？",
    ]
    st.caption("示例问题")
    cols = st.columns(len(sample_questions))
    for idx, sample in enumerate(sample_questions):
        if cols[idx].button(f"Q{idx+1}", key=f"sample_{idx}"):
            st.session_state.question_input = sample

    question = st.text_input(
        "请输入问题",
        placeholder="例如：键的抗拉强度最低要求是多少？",
        key="question_input",
    )

    if st.button("提交问题", type="primary") and question.strip():
        if not settings.chunks_path.exists():
            st.error("请先完成「解析全部 PDF」和「构建统一索引」")
        elif indexed_docs and not active_doc_ids:
            st.error("请至少选择一个问答范围文档")
        else:
            agent = get_agent()
            with st.spinner("检索与生成中..."):
                response = agent.ask(
                    question.strip(),
                    doc_ids=active_doc_ids or None,
                )

            scope = ", ".join(response.doc_ids) if response.doc_ids else "全部文档"
            st.markdown(f"**检索范围**: `{scope}` | **问题类型**: `{response.query_type}`")
            if response.status == "answered":
                st.success(response.answer)
            else:
                st.warning(response.refuse_reason or "已拒答")

            st.subheader("引用来源")
            if response.citations:
                for idx, citation in enumerate(response.citations, start=1):
                    st.markdown(
                        f"**[{idx}]** 文档 `{citation.doc_id or '-'}` | 第 {citation.page} 页 | "
                        f"条款 {citation.clause_id or '-'}  \n"
                        f"> {citation.quote}"
                    )
            else:
                st.write("无引用")

            st.subheader("检索证据")
            for hit in response.retrieved_chunks:
                with st.expander(
                    f"doc={hit.metadata.get('doc_id')} | score={hit.score:.3f} | "
                    f"page={hit.metadata.get('page')} | {hit.chunk_id}"
                ):
                    st.write(hit.text)

            st.subheader("自检结果")
            if response.self_check:
                st.json(
                    {
                        "grounding_level": response.self_check.grounding_level,
                        "grounding_score": response.self_check.grounding_score,
                        "citation_valid": response.self_check.citation_valid,
                        "numeric_valid": response.self_check.numeric_valid,
                        "checks_passed": response.self_check.checks_passed,
                        "checks_failed": response.self_check.checks_failed,
                    }
                )
