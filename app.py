import random
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
    is_rejected_answer,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
)
from quiz_core import generate_questions


# ========== 辅助函数（必须在最前面定义） ==========
def _get_next_question(rag, quiz):
    """
    获取下一道题。
    优先级：
    1. pending_queue 里还有题 → 直接取
    2. 从未用过的片段中随机选一个 → LLM 出题
    3. 全部片段用尽 → 返回 None
    """
    if quiz["pending_queue"]:
        quiz["total_asked"] += 1
        return quiz["pending_queue"].pop(0)

    total = len(rag.chunks)
    used = quiz["used_chunks"] | quiz["skipped_chunks"]
    available = [i for i in range(total) if i not in used]

    if not available:
        return None

    max_tries = 5
    for _ in range(max_tries):
        if not available:
            return None
        idx = random.choice(available)
        available.remove(idx)

        chunk = rag.chunks[idx]
        questions = generate_questions(chunk.page_content)

        if questions:
            quiz["used_chunks"].add(idx)
            quiz["pending_queue"] = questions[1:]
            quiz["total_asked"] += 1
            return questions[0]
        else:
            quiz["skipped_chunks"].add(idx)

    return None


def init_quiz_state():
    """初始化提问模式的状态"""
    return {
        "phase": "idle",
        "current": None,
        "user_answer": "",
        "pending_queue": [],
        "used_chunks": set(),
        "skipped_chunks": set(),
        "history": [],
        "total_asked": 0,
        "confirm_reset": False,
        "hints_shown": set(),
    }


# ========== 页面配置 ==========
st.set_page_config(page_title="智能文档问答系统", page_icon="📚", layout="wide")

# 固定标题的 CSS
st.markdown(
    """
    <style>
    .fixed-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: var(--background-color);
        padding: 0.5rem 0 0.5rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
st.markdown("## 📚 智能文档问答系统")
st.markdown('</div>', unsafe_allow_html=True)


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
    ("processed_sig", None),
    ("quiz", None),
    ("mode", "问答模式"),
    ("params_changed", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ========== 侧边栏 ==========
with st.sidebar:
    # 模式切换（放最上面）
    st.header("🎯 模式切换")
    st.session_state.mode = st.radio(
        "选择模式",
        ["📖 问答模式", "✏️ 提问模式"],
        index=0 if st.session_state.mode == "问答模式" else 1,
        label_visibility="collapsed",
    )
    # 同步状态
    if st.session_state.mode == "📖 问答模式":
        st.session_state.mode = "问答模式"
    else:
        st.session_state.mode = "提问模式"

    st.divider()

    # 文档上传
    st.header("📄 文档上传")
    uploaded = st.file_uploader(
        "支持 PDF / Markdown / TXT / Word（.docx）",
        type=["pdf", "md", "txt", "docx", "doc"],
    )

    st.header("⚙️ 参数设置")

    split_mode_label = st.radio(
        "切片方式",
        ["固定切片（快）", "动态切片（LLM 辅助，更准）"],
        index=0,
        help=(
            "固定切片：按字数机械切分，速度快。\n\n"
            "动态切片：调用 LLM 按语义边界切分，切片更聚焦主题，"
            "但预处理时需要额外等待约 10~30 秒。"
        ),
    )
    split_mode = "dynamic" if split_mode_label.startswith("动态") else "fixed"

    if split_mode == "fixed":
        chunk_size = st.slider("切片大小", 200, 1000, DEFAULT_CHUNK_SIZE, 50)
        chunk_overlap = st.slider("切片重叠", 0, 200, DEFAULT_CHUNK_OVERLAP, 10)
    else:
        chunk_size = DEFAULT_CHUNK_SIZE
        chunk_overlap = DEFAULT_CHUNK_OVERLAP

    top_k = st.slider("初检召回数量", 3, 20, DEFAULT_TOP_K)

    st.divider()

    def make_sig(uploaded, split_mode, chunk_size, chunk_overlap, top_k):
        if uploaded is None:
            return None
        return (
            f"{uploaded.name}|{uploaded.size}|{split_mode}|"
            f"{chunk_size}|{chunk_overlap}|{top_k}"
        )

    current_sig = make_sig(uploaded, split_mode, chunk_size, chunk_overlap, top_k)
    is_processed = (
        uploaded is not None
        and st.session_state.processed_sig is not None
        and st.session_state.processed_sig == current_sig
    )
    # 判断：参数是否变更但未重新处理
    params_changed = (
        uploaded is not None
        and st.session_state.processed_sig is not None
        and st.session_state.processed_sig != current_sig
    )
    # 存到 session_state，供主区域访问
    st.session_state.params_changed = params_changed

    if uploaded is None:
        st.button(
            "🚀 开始预处理",
            disabled=True,
            use_container_width=True,
            help="请先上传文档",
        )
        st.info("尚未上传文档")
    elif is_processed:
        st.button("✅ 处理完成", disabled=True, use_container_width=True)
        st.success(f"📌 当前文档源：{st.session_state.doc_name}")
        if st.session_state.rag is not None:
            with st.expander("🔍 当前检索器诊断", expanded=False):
                chunks = st.session_state.rag.chunks
                st.caption(f"片段总数：{len(chunks)}")
                st.text(chunks[0].page_content[:120])
    else:
        if st.button(
            "🚀 开始预处理",
            type="primary",
            use_container_width=True,
        ):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def on_progress(p, msg):
                progress_bar.progress(min(p / 100.0, 1.0))
                status_text.text(f"[{p}%] {msg}")

            status_text.text("[0%] 正在读取文档...")
            docs, error = load_uploaded_document(uploaded)

            if error:
                progress_bar.empty()
                status_text.empty()
                st.session_state.rag = None
                st.session_state.doc_name = None
                st.session_state.processed_sig = None
                st.error(error)
            else:
                rag = RAGSystem(
                    docs,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    split_mode=split_mode,
                    progress_callback=on_progress if split_mode == "dynamic" else None,
                )
                progress_bar.empty()
                status_text.empty()

                st.session_state.rag = rag
                st.session_state.doc_name = uploaded.name
                st.session_state.processed_sig = current_sig
                st.session_state.quiz = init_quiz_state()

                st.toast(
                    f"✅ 处理完成：{uploaded.name}"
                    f"（共 {len(rag.chunks)} 个片段）",
                    icon="✅",
                )
                st.rerun()

        st.info("参数已变更，请点击上方按钮重新处理文档")

    st.divider()

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# 主区域：根据模式分发
# ============================================================
mode = st.session_state.mode


# ============================================================
# 问答模式
# ============================================================
if mode == "问答模式":
    if st.session_state.rag is None:
        st.info("👈 请先在左侧上传文档并点击「开始预处理」")
    elif st.session_state.get("params_changed", False):
        st.warning(
            "⚠️ **参数已变更，请先重新处理文档**\n\n"
            "让新参数生效后再提问。"
        )
        # 仍然展示历史对话，但不显示输入框
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and msg.get("doc_name"):
                    st.caption(f"📄 本回答基于文档：**{msg['doc_name']}**")
                st.markdown(msg["content"])
                if msg.get("sources"):
                    st.markdown(f"📎 **引用来源（{len(msg['sources'])} 个片段）**")
                    for i, src in enumerate(msg["sources"]):
                        with st.expander(
                            f"[{i+1}] {src['source']}　—　相关度 {src['score']:.3f}",
                            expanded=False,
                        ):
                            st.text(src["content"])
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and msg.get("doc_name"):
                    st.caption(f"📄 本回答基于文档：**{msg['doc_name']}**")
                st.markdown(msg["content"])
                if msg.get("sources"):
                    st.markdown(f"📎 **引用来源（{len(msg['sources'])} 个片段）**")
                    for i, src in enumerate(msg["sources"]):
                        with st.expander(
                            f"[{i+1}] {src['source']}　—　相关度 {src['score']:.3f}",
                            expanded=False,
                        ):
                            st.text(src["content"])

        if query := st.chat_input("请输入你的问题..."):
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
                    rewritten = rewrite_query(
                        query, st.session_state.messages[:-1], current_doc
                    )
                    if rewritten != query:
                        st.caption(f"🔍 改写后的问题：{rewritten}")

                    stream, top_docs, top_scores = rag.stream_answer(
                        query=query,
                        retrieve_query=rewritten,
                        use_rerank=True,
                    )
                    answer = st.write_stream(stream)

                    if is_rejected_answer(answer):
                        st.info("💡 文档中无相关内容，未提供引用")
                        sources = []
                    else:
                        sources = [
                            {
                                "source": d.metadata.get("source", "未知"),
                                "score": s,
                                "content": d.page_content,
                            }
                            for d, s in zip(top_docs, top_scores)
                        ]
                        if sources:
                            st.markdown(f"📎 **引用来源（{len(sources)} 个片段）**")
                            for i, src in enumerate(sources):
                                with st.expander(
                                    f"[{i+1}] {src['source']}　—　相关度 {src['score']:.3f}",
                                    expanded=False,
                                ):
                                    st.text(src["content"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "doc_name": current_doc,
            })


# ============================================================
# 提问模式
# ============================================================
elif mode == "提问模式":
    if st.session_state.rag is None:
        st.info("👈 请先在左侧上传文档并点击「开始预处理」")
    elif st.session_state.get("params_changed", False):
        st.warning(
            "⚠️ **参数已变更，请先重新处理文档**\n\n"
            "让新参数生效后再使用提问模式。"
        )
        st.info("💡 重新处理会重置提问进度（已出题记录、历史记录），请知悉。")
    else:
        rag = st.session_state.rag
        total_chunks = len(rag.chunks)

        if st.session_state.quiz is None:
            st.session_state.quiz = init_quiz_state()

        quiz = st.session_state.quiz

        # 顶部进度卡片
        covered = len(quiz["used_chunks"]) + len(quiz["skipped_chunks"])
        col1, col2, col3 = st.columns(3)
        col1.metric("已出题", f"{quiz['total_asked']} 道")
        col2.metric("文档覆盖", f"{covered} / {total_chunks} 片段")
        col3.metric("剩余可出题片段", max(0, total_chunks - covered))

        st.divider()

        # ====== 状态 1：未开始 ======
        if quiz["phase"] == "idle":
            st.markdown("### ✏️ 提问模式")
            st.markdown(
                "系统将从文档中选取内容并出题，你凭记忆回答后，"
                "会看到参考答案和答案依据。\n\n"
                "**只出简答题，不做判分**——你对照答案自我评估即可。"
            )
            if st.button("🚀 开始提问", type="primary", use_container_width=True):
                quiz["phase"] = "generating"
                st.rerun()

        # ====== 状态 2：出题中 ======
        elif quiz["phase"] == "generating":
            with st.spinner("正在从文档中选取内容并生成题目..."):
                question = _get_next_question(rag, quiz)
                if question is None:
                    quiz["phase"] = "all_covered"
                else:
                    quiz["current"] = question
                    quiz["user_answer"] = ""
                    quiz["phase"] = "asking"
                st.rerun()

        # ====== 状态 3：等待作答 ======
        elif quiz["phase"] == "asking":
            q = quiz["current"]
            st.markdown(f"### 📝 第 {quiz['total_asked']} 题")
            st.markdown(q["question"])
            st.markdown("")

            user_answer = st.text_area(
                "你的答案：",
                value=quiz["user_answer"],
                height=120,
                placeholder="请输入你的答案，或点击下方「跳过此题」...",
                key=f"answer_input_{quiz['total_asked']}",
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ 提交", type="primary", use_container_width=True):
                    quiz["user_answer"] = user_answer or "（未作答）"
                    quiz["phase"] = "showing"
                    st.rerun()
            with col2:
                if st.button("⏭️ 跳过此题", use_container_width=True):
                    quiz["user_answer"] = "（用户跳过）"
                    quiz["phase"] = "showing"
                    st.rerun()

        # ====== 状态 4：展示答案 ======
        elif quiz["phase"] == "showing":
            q = quiz["current"]
            st.markdown(f"### 📝 第 {quiz['total_asked']} 题")
            st.markdown(q["question"])

            st.markdown("---")
            st.markdown("**你的答案：**")
            st.info(quiz["user_answer"])

            st.markdown("**✅ 参考答案：**")
            st.success(q["reference_answer"])

            st.markdown("**📎 答案依据：**")
            st.markdown(f"> {q['reference_source']}")

            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("➡️ 下一题", type="primary", use_container_width=True):
                    quiz["history"].append({
                        "question": q["question"],
                        "user_answer": quiz["user_answer"],
                        "reference_answer": q["reference_answer"],
                        "reference_source": q["reference_source"],
                    })
                    quiz["current"] = None
                    quiz["user_answer"] = ""
                    quiz["phase"] = "generating"
                    st.rerun()
            with col2:
                if st.button("🛑 结束测验", use_container_width=True):
                    quiz["history"].append({
                        "question": q["question"],
                        "user_answer": quiz["user_answer"],
                        "reference_answer": q["reference_answer"],
                        "reference_source": q["reference_source"],
                    })
                    quiz["current"] = None
                    quiz["user_answer"] = ""
                    quiz["phase"] = "summary"
                    st.rerun()

        # ====== 状态 5：总结 ======
        elif quiz["phase"] == "summary":
            st.markdown("### 🎉 本次测验结束")
            st.markdown(f"**共完成 {len(quiz['history'])} 道题**")

            if len(quiz["history"]) in (20, 50) and len(quiz["history"]) not in quiz["hints_shown"]:
                quiz["hints_shown"].add(len(quiz["history"]))
                st.info(f"💡 你已经答了很多题了（{len(quiz['history'])} 道），可以随时结束。")

            st.markdown("---")
            st.markdown("#### 📋 本次提问历史")
            if not quiz["history"]:
                st.info("本次测验还没有记录。")
            else:
                for i, item in enumerate(quiz["history"], 1):
                    with st.expander(f"第 {i} 题：{item['question'][:40]}...", expanded=False):
                        st.markdown(f"**题目：** {item['question']}")
                        st.markdown(f"**你的答案：** {item['user_answer']}")
                        st.markdown(f"**参考答案：** {item['reference_answer']}")
                        st.markdown(f"**答案依据：** > {item['reference_source']}")

            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 开始新一轮", type="primary", use_container_width=True):
                    quiz["history"] = []
                    quiz["phase"] = "generating"
                    st.rerun()
            with col2:
                if st.button("♻️ 重置进度", use_container_width=True):
                    quiz["confirm_reset"] = True
                    st.rerun()

            if quiz["confirm_reset"]:
                st.warning("⚠️ 重置进度将清空已出题记录和跳过记录，从头开始。确定吗？")
                cc1, cc2 = st.columns([1, 1])
                with cc1:
                    if st.button("✅ 确定重置", type="primary", use_container_width=True):
                        st.session_state.quiz = init_quiz_state()
                        st.rerun()
                with cc2:
                    if st.button("❌ 取消", use_container_width=True):
                        quiz["confirm_reset"] = False
                        st.rerun()

        # ====== 状态 6：全部覆盖 ======
        elif quiz["phase"] == "all_covered":
            st.markdown("### 🎉 文档内容已全部覆盖")
            st.markdown(
                f"共出题 **{quiz['total_asked']}** 道，"
                f"已遍历文档的 {total_chunks} 个片段。"
            )
            st.info("所有片段都出过题了。你可以重置进度，重新开始。")

            st.markdown("---")
            st.markdown("#### 📋 本次提问历史")
            for i, item in enumerate(quiz["history"], 1):
                with st.expander(f"第 {i} 题：{item['question'][:40]}...", expanded=False):
                    st.markdown(f"**题目：** {item['question']}")
                    st.markdown(f"**你的答案：** {item['user_answer']}")
                    st.markdown(f"**参考答案：** {item['reference_answer']}")

            st.markdown("---")
            if st.button("♻️ 重置进度，重新开始", type="primary", use_container_width=True):
                st.session_state.quiz = init_quiz_state()
                st.rerun()