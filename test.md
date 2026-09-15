# RAG 项目测试文档

## 什么是 RAG

RAG 全称 Retrieval-Augmented Generation，中文叫检索增强生成。
它的核心思想是：先根据用户问题检索相关文档片段，再让大模型基于这些片段生成答案。
核心流程是：用户问题 → 检索相关文档 → 将文档作为上下文 → 大模型生成答案。

## RAG 的优势

1. 可以减少大模型的幻觉问题。
2. 可以让大模型回答私有文档中的内容。
3. 答案可以附带引用来源，方便溯源。
4. 不需要重新训练模型，只需更新知识库。

## RAG 的典型应用场景

- 企业知识库问答
- 法律/医疗文档查询
- 客服工单自动回复
- 个人笔记助手

## 本项目技术栈

- 开发语言：Python 3.11
- 框架：LangChain 1.x
- 向量库：Chroma 1.5.9
- Embedding：BAAI/bge-small-zh-v1.5
- 大模型：DeepSeek Chat
- 前端：Streamlit
- 评估：RAGAS

## 为什么选 Chroma

Chroma 安装简单，与 LangChain 深度集成，支持本地持久化，非常适合个人项目和快速 Demo。
它不需要单独起服务，pip install 之后就能用。

## 为什么选 BGE Embedding

BGE 系列是智源研究院开源的中文 Embedding 模型，中文语义效果优于通用模型。
bge-small-zh-v1.5 体积小、速度快，适合个人电脑 CPU 运行。

## 为什么选 DeepSeek

DeepSeek 是国产大模型 API，价格便宜、国内可直连，兼容 OpenAI 接口格式，
非常适合开发阶段快速迭代。

## 混合检索的作用

混合检索 = 向量检索 + BM25 关键词检索。
向量检索擅长语义相似，BM25 擅长关键词精确匹配，两者结合能提升召回率。
权重通常设置为 BM25 0.4、向量 0.6，可根据语料调整。

## Rerank 的作用

Rerank 是对初步召回的 Top-K 结果做二次精排。
初检可以用快模型，Rerank 用更精细的 CrossEncoder 模型，
但只对少量候选做计算，总体延迟可控。