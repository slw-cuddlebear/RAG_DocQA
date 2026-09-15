import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from rag_core import (
    RAGSystem,
    load_uploaded_document,
    rewrite_query,
    get_embeddings,
    get_reranker,
    get_llm,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_RERANK_TOP_K,
)


# ========== 页面配置 ==========
st.set_page_config(page_title="智能文档问答系统", page_icon="📚", layout="wide")
st.title("📚 智能文档问答系统")
st.caption(
    "上传 PDF / Markdown / TXT / Word(.docx)，用自然语言提问，"
    "系统基于文档内容回答并给出引用来源。"
    "（不支持图片型 PDF 和旧版 .doc）"
)


# ========== 缓存模型 ==========
@st.cache_resource(show_spinner="正在加载模型（首次较慢，请耐心等待）...")
def warmup_models():
    get_embeddings()
    get_reranker()
    get_llm()
    return True


warmup_models()


# ========== 会话状态 ==========
for key, default in [
    ("messages", []),
    ("rag", None),
    ("doc_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ========== 侧边栏 ==========
with st.sidebar:
    st.header("📄 文档上传")
    uploaded = st.file_uploader(
        "支持 PDF / Markdown / TXT / Word（.docx）",
        type=["pdf", "md", "txt", "docx", "doc"],
    )

    if st.session_state.doc_name:
        st.success(f"📌 当前文档源：{st.session_state.doc_name}")
    else:
        st.info("尚未上传文档")

    # 诊断信息：确认 retriever 里装的是哪份文档
    if st.session_state.rag is not None:
        with st.expander("🔍 当前检索器诊断", expanded=False):
            chunks = st.session_state.rag.chunks
            st.caption(f"片段总数：{len(chunks)}")
            st.caption(f"首片段来源：{chunks[0].metadata.get('source', '未知')}")
            st.text(chunks[0].page_content[:120])

    st.header("⚙️ 参数设置")
    chunk_size = st.slider("切片大小", 200, 1000, DEFAULT_CHUNK_SIZE, 50)
    chunk_overlap = st.slider("切片重叠", 0, 200, DEFAULT_CHUNK_OVERLAP, 10)
    top_k = st.slider("初检召回数量", 3, 20, DEFAULT_TOP_K)
    rerank_top_k = st.slider("Rerank 后保留数量", 1, 5, DEFAULT_RERANK_TOP_K)

    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()


# ========== 文档上传处理 ==========
if uploaded is not None:
    if st.session_state.doc_name != uploaded.name:
        with st.spinner(f"正在处理 {uploaded.name} ..."):
            docs, error = load_uploaded_document(uploaded)
            if error:
                st.session_state.rag = None
                st.session_state.doc_name = None
                st.error(error)
                st.rerun()
            else:
                rag = RAGSystem(docs, chunk_size, chunk_overlap, top_k)
                old_name = st.session_state.doc_name
                st.session_state.rag = rag
                st.session_state.doc_name = uploaded.name
                # 保留历史对话，不做清空

                if old_name is None:
                    st.toast(
                        f"✅ 文档已加载：{uploaded.name}"
                        f"（共 {len(rag.chunks)} 个片段）",
                        icon="✅",
                    )
                else:
                    st.toast(
                        f"🔄 文档源已更换为：{uploaded.name}"
                        f"（共 {len(rag.chunks)} 个片段）",
                        icon="🔄",
                    )
                st.rerun()
else:
    if st.session_state.doc_name is not None:
        st.session_state.rag = None
        st.session_state.doc_name = None
        st.toast("🗑️ 文档已移除", icon="🗑️")
        st.rerun()


# ========== 显示对话历史 ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("doc_name"):
            st.caption(f"📄 本回答基于文档：**{msg['doc_name']}**")
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 引用来源"):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(
                        f"**[{i+1}] {src['source']}** "
                        f"(score: {src['score']:.3f})"
                    )
                    st.text(src["preview"])


# ========== 问题输入 ==========
if query := st.chat_input("请输入你的问题..."):
    if st.session_state.rag is None:
        st.warning("请先在左侧上传文档")
    else:
        current_doc = st.session_state.doc_name
        rag = st.session_state.rag

        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "doc_name": current_doc,
        })
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            st.caption(f"📄 本回答基于文档：**{current_doc}**")

            with st.spinner("正在检索和生成..."):
                # 改写查询（只取当前文档的历史）
                rewritten = rewrite_query(
                    query, st.session_state.messages[:-1], current_doc
                )
                if rewritten != query:
                    st.caption(f"🔍 改写后的问题：{rewritten}")

                # 检索 + Rerank + 流式生成
                stream, top_docs, top_scores = rag.stream_answer(
                    query=query,
                    retrieve_query=rewritten,
                    use_rerank=True,
                    rerank_top_k=rerank_top_k,
                )
                answer = st.write_stream(stream)

                # 引用来源
                sources = [
                    {
                        "source": d.metadata.get("source", "未知"),
                        "score": s,
                        "preview": d.page_content[:200],
                    }
                    for d, s in zip(top_docs, top_scores)
                ]
                if sources:
                    with st.expander("📎 引用来源"):
                        for i, src in enumerate(sources):
                            st.markdown(
                                f"**[{i+1}] {src['source']}** "
                                f"(score: {src['score']:.3f})"
                            )
                            st.text(src["preview"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "doc_name": current_doc,
        })