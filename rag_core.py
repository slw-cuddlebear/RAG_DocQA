"""
RAG 核心模块：模型管理、文档加载、切片、混合检索、Rerank、生成
供 app.py / eval_*.py 共用，保证配置一致。
"""
import os
import re
import sys
import types
import uuid
import tempfile
import warnings

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)

# RAGAS 0.4.3 兼容垫片（必须在 import ragas 之前执行）
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vx = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:
        pass

    _vx.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vx

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder


# ========== 默认参数 ==========
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 20
DEFAULT_RERANK_TOP_K = 3
DEFAULT_ENSEMBLE_WEIGHTS = [0.4, 0.6]

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "deepseek-chat"
LLM_BASE_URL = "https://api.deepseek.com"


# ========== 模型单例（进程内共享） ==========
_embeddings = None
_reranker = None
_llm = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=LLM_BASE_URL,
            temperature=0.3,
        )
    return _llm


# ========== 文本清洗 ==========
def clean_text(text: str) -> str:
    """去掉 Word 文档里残留的 LaTeX 标记，例如 \\(90\\%\\) → 90%"""
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text)
    text = text.replace("\\%", "%")
    return text


# ========== 文档加载 ==========
def load_local_document(file_path: str, source_name: str = None) -> list:
    """从本地路径加载文档，支持 .pdf / .md / .txt / .docx"""
    suffix = file_path.rsplit(".", 1)[-1].lower()
    if suffix == "pdf":
        loader = PyPDFLoader(file_path)
    elif suffix in ("md", "txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif suffix == "docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"不支持的文件类型: .{suffix}")

    docs = loader.load()
    name = source_name or file_path
    for d in docs:
        d.metadata["source"] = name
    return docs


def load_uploaded_document(uploaded_file):
    """
    从 Streamlit 上传对象加载文档。
    返回 (docs, error_msg)：成功时 error_msg 为 None。
    """
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if suffix == "doc":
        return [], (
            "❌ 不支持旧版 .doc 格式。\n\n"
            "请用 Word 打开该文件 → 文件 → 另存为 → "
            "保存类型选 “Word 文档 (*.docx)” → 再上传。"
        )
    if suffix not in ("pdf", "md", "txt", "docx"):
        return [], f"❌ 不支持的文件类型：.{suffix}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        docs = load_local_document(tmp_path, source_name=uploaded_file.name)
    except Exception as e:
        return [], f"❌ 文档解析失败：{e}"

    total_text = sum(len(d.page_content.strip()) for d in docs)
    if total_text < 50:
        return [], (
            "❌ 文档内容太少或无法提取文字。\n\n"
            "如果是 PDF，可能为扫描件（图片型 PDF）；如果是 Word，可能内容过少。"
        )

    return docs, None


# ========== Prompt 模板 ==========
RAG_PROMPT = ChatPromptTemplate.from_template(
    """你是一个严谨的文档问答助手。请根据下面提供的文档片段回答用户问题。

要求：
1. 优先根据文档片段回答。
2. 可以在文档片段的基础上做合理归纳和推断，但不要引入文档之外的背景知识。
3. 只有在文档完全没有相关内容时，才回答"根据现有资料无法回答"。
4. 回答要简洁、准确。
5. 如果文档中提到的是"目标值/计划值/预期值"等未实现的数值，
   而用户问的是"实际达成/已经完成"的数值，
   请明确说明"文档中提到的是目标值，并非实际达成数据"，然后给出目标值。

文档片段：
{context}

用户问题：{question}

答案："""
)

REWRITE_PROMPT = """你是一个查询改写助手。请根据下面的对话历史，把用户的最新问题改写成一个独立的、包含完整信息的问题。

规则：
1. 如果最新问题已经完整，直接返回原问题。
2. 如果最新问题包含"它""这个""那"等指代词，根据历史补全。
3. 只返回改写后的问题，不要解释。

对话历史：
{history}

最新问题：{question}

改写后的问题："""


def format_docs(docs) -> str:
    """把检索到的文档片段格式化成 prompt 里的 context"""
    return "\n\n".join(
        f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
        for d in docs
    )


def rewrite_query(question: str, messages: list, current_doc: str) -> str:
    """根据历史对话改写查询，只取属于当前文档的历史"""
    if not messages:
        return question
    relevant = [m for m in messages if m.get("doc_name") == current_doc]
    if not relevant:
        return question
    history = "\n".join(
        f"{m['role']}: {m['content'][:200]}" for m in relevant[-6:]
    )
    return get_llm().invoke(
        REWRITE_PROMPT.format(history=history, question=question)
    ).content.strip()


# ========== RAG 系统 ==========
class RAGSystem:
    """
    统一的 RAG 系统。
    - 初始化：切片（固定 / 动态）→ 清洗 → 向量化 → 构建混合检索器
    - retrieve()：检索 Top-K 片段
    - answer()：检索 + 生成完整答案
    - stream_answer()：检索 + 流式生成
    """

    def __init__(
        self,
        docs,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        weights: list = None,
        split_mode: str = "fixed",
        progress_callback=None,
    ):
        """
        split_mode:
        - "fixed"   : 固定长度切片（RecursiveCharacterTextSplitter）
        - "dynamic" : LLM 辅助动态切片（按语义边界切分）
        """
        if split_mode == "dynamic":
            from dynamic_split import dynamic_split
            self.chunks = dynamic_split(
                docs, progress_callback=progress_callback
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            self.chunks = splitter.split_documents(docs)

        # 清洗 LaTeX 标记（两种切片方式都需要）
        for d in self.chunks:
            d.page_content = clean_text(d.page_content)

        embeddings = get_embeddings()

        # 每次用唯一的 collection_name，避免 Chroma 同进程内复用默认 collection
        collection_name = f"rag_{uuid.uuid4().hex[:12]}"

        vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            collection_name=collection_name,
        )
        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )

        bm25 = BM25Retriever.from_documents(self.chunks)
        bm25.k = top_k

        self.retriever = EnsembleRetriever(
            retrievers=[bm25, vector_retriever],
            weights=weights or DEFAULT_ENSEMBLE_WEIGHTS,
        )

    def retrieve(self, query: str, use_rerank: bool = True,
                 min_k: int = 1, max_k: int = 5):
        """
        动态 Top-K：
        - 根据 Rerank 分数自动决定引用几个片段
        - 至少 min_k 个，至多 max_k 个
        - 相对阈值 0.7（保留最高分 70% 以上的）
        - 断崖检测（相邻骤降 50% 时切断）
        """
        candidates = self.retriever.invoke(query)

        if not use_rerank:
            return candidates[:3], [0.0] * 3

        reranker = get_reranker()
        pairs = [(query, d.page_content) for d in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )

        if not ranked:
            return [], []

        # 相对阈值过滤
        max_score = ranked[0][1]
        threshold = max_score * 0.7
        filtered = [(d, s) for d, s in ranked if s >= threshold]

        # 断崖检测
        if len(filtered) > min_k:
            for i in range(len(filtered) - 1):
                curr, nxt = filtered[i][1], filtered[i + 1][1]
                if curr > 0 and nxt < curr * 0.5:
                    filtered = filtered[: i + 1]
                    break

        # 边界保护
        if len(filtered) < min_k:
            filtered = ranked[:min_k]
        filtered = filtered[:max_k]

        return [d for d, _ in filtered], [float(s) for _, s in filtered]

    def answer(
            self,
            query: str,
            retrieve_query: str = None,
            use_rerank: bool = True,
    ):
        """
        返回 (answer_text, top_docs, top_scores)
        - query：用于生成答案（传给 LLM）
        - retrieve_query：用于检索（默认与 query 相同）
        """
        retrieve_query = retrieve_query or query
        top_docs, top_scores = self.retrieve(retrieve_query, use_rerank)
        context = format_docs(top_docs)
        answer_text = get_llm().invoke(
            RAG_PROMPT.format(context=context, question=query)
        ).content
        return answer_text, top_docs, top_scores

    def stream_answer(
            self,
            query: str,
            retrieve_query: str = None,
            use_rerank: bool = True,
    ):
        """
        返回 (stream_iterator, top_docs, top_scores)
        """
        retrieve_query = retrieve_query or query
        top_docs, top_scores = self.retrieve(retrieve_query, use_rerank)
        context = format_docs(top_docs)
        stream = get_llm().stream(
            RAG_PROMPT.format(context=context, question=query)
        )
        return stream, top_docs, top_scores

# ========== 拒答检测 ==========
REJECT_PATTERNS = [
    "根据现有资料无法回答", "无法回答", "没有相关信息",
    "资料中没有", "未提及", "没有提到", "文档中未", "未说明",
]


def is_rejected_answer(answer: str) -> bool:
    """判断回答是否为拒答"""
    return any(p in answer for p in REJECT_PATTERNS)
