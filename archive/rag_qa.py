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

# ========== 1. 准备向量库 ==========
loader = TextLoader("../test.md", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=30
)
chunks = text_splitter.split_documents(documents)

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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ========== 2. 初始化 DeepSeek ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3
)

# ========== 3. 构建 Prompt ==========
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

# ========== 4. 组装 RAG 链 ==========
def format_docs(docs):
    return "\n\n".join(
        f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
        for d in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ========== 5. 测试 ==========
if __name__ == "__main__":
    questions = [
        "RAG 有什么优势？",
        "这个项目用了什么向量数据库？",
        "今天的天气怎么样？",   # 故意问一个文档里没有的，测试拒答
    ]

    for q in questions:
        print(f"\n{'='*50}")
        print(f"问题：{q}")
        print(f"{'='*50}")

        # 先取检索结果，方便展示来源
        docs = retriever.invoke(q)
        answer = rag_chain.invoke(q)

        print(f"\n答案：{answer}")
        print(f"\n引用来源：")
        for i, d in enumerate(docs):
            print(f"  [{i+1}] {d.metadata.get('source', '未知')}")