import os
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

# ========== 1. 加载 + 切片 ==========
from langchain_community.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("../bp.docx")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=30
)
chunks = text_splitter.split_documents(documents)
print(f"切片数：{len(chunks)}")

# ========== 2. 向量库 + BM25 + 融合 ==========
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 6

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

# ========== 3. Rerank 模型 ==========
print("正在加载 Rerank 模型...")
reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
print("Rerank 模型加载完成")

def rerank(query, docs, top_k=3):
    """对候选文档重新打分排序"""
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]], [float(s) for _, s in ranked[:top_k]]

# ========== 4. LLM ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_template(
    """你是一个严谨的文档问答助手。请根据下面提供的文档片段回答用户问题。

要求：
1. 只根据文档片段回答，不要编造。
2. 如果文档中没有相关信息，回答"根据现有资料无法回答"。
3. 回答要简洁、准确。

文档片段：
{context}

用户问题：{question}

答案："""
)

def format_docs(docs):
    return "\n\n".join(
        f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
        for d in docs
    )

# ========== 5. 手动执行 RAG 流程 ==========
def rag_answer(query):
    # 初检
    candidates = ensemble_retriever.invoke(query)
    # Rerank
    top_docs, top_scores = rerank(query, candidates, top_k=3)
    # 拒答阈值：如果最高分太低，直接拒答
    if top_scores and top_scores[0] < 0.0:
        return "根据现有资料无法回答", []

    context = format_docs(top_docs)
    answer = llm.invoke(prompt.format(context=context, question=query)).content
    return answer, top_docs

# ========== 6. 测试 ==========
if __name__ == "__main__":
    questions = [
        "RAG 有什么优势？",
        "这个项目用了什么向量数据库？",
        "Embedding 用的是什么模型？",
        "RAG 适合用在哪些场景？",
        "今天的天气怎么样？",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")

        answer, docs = rag_answer(q)

        print(f"\n答案：{answer}")
        print(f"\nRerank 后保留 {len(docs)} 个片段：")
        for i, d in enumerate(docs):
            preview = d.page_content[:70].replace("\n", " ")
            print(f"  [{i+1}] {preview}...")