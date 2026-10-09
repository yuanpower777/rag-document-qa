"""Streamlit 网页界面：上传文档 → 提问 → 带来源引用的回答。"""
from __future__ import annotations

import os
import tempfile

import streamlit as st

import rag

st.set_page_config(page_title="智能文档问答助手", page_icon="📚", layout="wide")

st.title("📚 智能文档问答助手")
st.caption("基于 RAG（检索增强生成）：先检索文档片段，再让大模型依据片段作答并标注来源，降低幻觉。")

with st.sidebar:
    st.header("参数设置")
    rag.CHUNK_SIZE = st.slider("切分粒度 chunk_size", 200, 1000, rag.CHUNK_SIZE, 50)
    rag.TOP_K = st.slider("召回条数 Top-K", 1, 8, rag.TOP_K)
    st.divider()
    st.write("**配置状态**")
    st.write(f"对话模型：`{rag.LLM_MODEL}`")
    st.write(f"向量模型：`{rag.EMBED_MODEL or '本地兜底向量化'}`")
    if not rag.LLM_API_KEY:
        st.error("未检测到 DEEPSEEK_API_KEY，生成回答会失败。")
    st.divider()
    if st.button("清空知识库与对话"):
        st.session_state.pop("store", None)
        st.session_state.pop("messages", None)
        st.rerun()

if "store" not in st.session_state:
    st.session_state.store = rag.VectorStore()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = []

store: rag.VectorStore = st.session_state.store

uploaded = st.file_uploader(
    "上传知识库文档（支持 .txt / .md / .pdf，可多选）",
    type=["txt", "md", "pdf", "docx"], accept_multiple_files=True)

if uploaded:
    pending = [f.name for f in uploaded if f.name not in st.session_state.indexed]
    if pending and st.button(f"建立索引（{len(pending)} 个新文件）"):
        with st.spinner("正在解析、切分并向量化..."):
            for item in uploaded:
                if item.name in st.session_state.indexed:
                    continue
                suffix = os.path.splitext(item.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(item.getbuffer())
                    tmp_path = tmp.name
                try:
                    text = rag.load_text(tmp_path)
                finally:
                    os.unlink(tmp_path)
                count = store.add_document(text, item.name)
                st.session_state.indexed.append(item.name)
                st.success(f"《{item.name}》切分为 {count} 个文本块并已入库")
        st.rerun()

st.caption(f"知识库：{len(store.chunks)} 个文本块，来自 {len(st.session_state.indexed)} 个文件")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for hit in msg.get("hits", []):
            with st.expander(f"来源：{hit.source}（相似度 {hit.score:.3f}）"):
                st.write(hit.text)

if prompt := st.chat_input("就你上传的文档提问，例如：这份文档的核心结论是什么？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not store.chunks:
            st.warning("请先上传文档并建立索引。")
        else:
            hits = store.search(prompt)
            try:
                placeholder = st.empty()
                buffer = ""
                for piece in rag.generate(prompt, hits, stream=True):
                    buffer += piece
                    placeholder.markdown(buffer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": buffer, "hits": hits})
            except Exception as exc:  # noqa: BLE001
                st.error(f"生成失败：{exc}")
