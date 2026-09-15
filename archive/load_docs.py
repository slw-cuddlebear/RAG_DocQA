import os
from dotenv import load_dotenv

# 加载 .env 里的环境变量
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 加载 Markdown 文档
loader = TextLoader("../test.md", encoding="utf-8")
documents = loader.load()
print(f"加载了 {len(documents)} 个文档对象")

# 2. 切片
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
chunks = text_splitter.split_documents(documents)
print(f"切成了 {len(chunks)} 个块")

# 3. 初始化 Embedding 模型
print("正在加载 Embedding 模型，第一次可能需要几分钟...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("Embedding 模型加载完成")

# 4. 存入 Chroma
print("正在向量化并存入 Chroma...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"已存入 Chroma，共 {vectorstore._collection.count()} 条向量")

# 5. 简单测试检索功能
query = "RAG 有什么优势？"
results = vectorstore.similarity_search(query, k=2)
print(f"\n查询：{query}")
for i, doc in enumerate(results):
    print(f"\n--- 结果 {i+1} ---")
    print(doc.page_content)
    print("来源:", doc.metadata.get("source"))