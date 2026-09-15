# 📚 RAG 智能文档问答系统
![img.png](img.png)

基于 RAG（检索增强生成）的文档问答系统。上传文档，用自然语言提问，系统检索相关片段后调用大模型生成带引用来源的答案。

## ✨ 功能特性

- 📄 支持 PDF / Markdown / TXT / Word(.docx) 上传
- 🔍 混合检索（BM25 关键词 + 向量语义）
- 🎯 CrossEncoder Rerank 精排
- 💬 多轮对话（支持指代追问，如"它的定价呢？"）
- 📎 引用来源溯源（显示片段来源和 Rerank 分数）
- 🚫 拒答机制（文档无相关内容时主动拒答）
- ⚡ 流式输出（答案逐字生成）
- 📊 RAGAS 量化评估 + 消融实验

## 🏗️ 系统架构

```
用户提问
    ↓
查询改写（多轮对话时，将"它"改写为完整问题）
    ↓
混合检索（BM25 + 向量检索，权重 0.4 : 0.6）
    ↓
Rerank 精排（CrossEncoder，Top-15 → Top-3）
    ↓
构建 Prompt（问题 + 检索片段）
    ↓
LLM 生成答案（DeepSeek Chat，流式输出）
    ↓
返回答案 + 引用来源
```

## 🛠️ 技术栈

| 模块 | 方案 | 说明 |
|---|---|---|
| 开发语言 | Python 3.11 | |
| RAG 框架 | LangChain 1.x | |
| 向量数据库 | Chroma | 内存模式，每次上传文档独立隔离 |
| Embedding | BAAI/bge-small-zh-v1.5 | 中文效果好，CPU 可跑 |
| Rerank | BAAI/bge-reranker-base | CrossEncoder 精排 |
| 大模型 | DeepSeek Chat | 兼容 OpenAI 接口，便宜稳定 |
| 前端 | Streamlit | 快速出 Demo |
| 文档解析 | PyPDF2 / docx2txt | |
| 评估 | RAGAS | 四维度量化评估 |

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/RAG_DocQA.git
cd RAG_DocQA
```

### 2. 创建虚拟环境

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，并填入你的 Key：

```bash
cp .env.example .env
```

> DeepSeek API Key 申请地址：https://platform.deepseek.com

### 5. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

## 📖 使用说明

1. 在左侧侧边栏上传文档（支持 PDF / Markdown / TXT / docx）
2. 调整参数（可选）：
   - 切片大小：默认 600
   - 切片重叠：默认 100
   - 初检召回数量：默认 15
   - Rerank 后保留数量：默认 3
3. 在底部输入框提问
4. 系统返回答案 + 引用来源

### 注意事项

- ❌ 不支持图片型 PDF（扫描件），因为无法提取文字层
- ❌ 不支持旧版 `.doc` 格式，请先另存为 `.docx`
- ✅ 支持换文档时保留历史对话，每次回答上方标注"基于文档：xxx"

## 📊 评估结果

### RAGAS 主评估（30 道正向问题，含 Rerank）

| 指标 | 得分 | 说明 |
|---|---|---|
| Faithfulness | **0.9933** | 答案忠于原文，无编造 |
| Answer Relevancy | **0.9312** | 答案与问题高度相关 |
| Context Precision | **0.7556** | 检索结果精确度 |
| Context Recall | **0.8722** | 检索结果召回率 |

### 消融实验（验证 Rerank 价值）

| 指标 | 加 Rerank | 不加 Rerank | 提升 |
|---|---|---|---|
| Faithfulness | **0.9933** | 0.7504 | **+0.24** |
| Answer Relevancy | **0.9312** | 0.4773 | **+0.45** |
| Context Precision | **0.7556** | 0.1833 | **+0.57** |
| Context Recall | **0.8722** | 0.3228 | **+0.55** |

> 在 37 页文档（134 个片段）上，不加 Rerank 时 30 道题有 18 道直接拒答；加 Rerank 后全部正常回答。说明在大文档场景下，CrossEncoder 精排是必要环节。

### 拒答测试（15 道负向问题）

| 类别 | 准确率 |
|---|---|
| 文档未提及实体 | 4/5 |
| 文档未提及数字 | 3/5 |
| 边缘模糊问题 | 4/5 |
| **总准确率** | **73.33%** |

> 4 道"失败"的题目均为部分回答（如文档有目标值但无实际值），并非模型幻觉。

### 评估工具说明

RAGAS 评估使用 DeepSeek 作为裁判，存在同源偏见。RAGAS 通过拆解任务（文本蕴含判断、反向生成问题）降低主观性，但无法完全消除。未来可引入 GPT-4o 或 Qwen 作为独立裁判进行交叉验证。

## 📁 项目结构

```
RAG_DocQA/
├── rag_core.py                     # RAG 核心逻辑（模型管理、检索、生成）
├── app.py                          # Streamlit 前端
├── eval_rag.py                     # RAGAS 评估脚本（支持 --no-rerank）
├── eval_reject.py                  # 拒答能力评估脚本
├── eval_questions.json             # 30 道正向测试题
├── reject_questions.json           # 15 道负向测试题
├── requirements.txt                # Python 依赖
├── test.md                         # 示例文档（小）
├── bp.docx                         # 示例文档（大，37 页）
├── .env                            # 环境变量（含密钥，不提交）
├── .env.example                    # 环境变量模板（提交，需自行配置key）
├── .gitignore
├── eval_result_with_rerank.csv     # 评估结果
├── eval_result_no_rerank.csv
├── eval_reject_result.csv
└── archive/                        # 历史版本（Phase1-3 脚本）
    ├── load_docs.py
    ├── rag_qa.py
    ├── rag_qa_hybrid.py
    └── rag_qa_rerank.py
```

## 🔬 运行评估

```bash
# RAGAS 评估（加 Rerank）
python eval_rag.py

# 消融实验（不加 Rerank）
python eval_rag.py --no-rerank

# 拒答测试
python eval_reject.py
```

## 🎯 已实现 / 待优化

### 已实现

- [x] 多格式文档加载（PDF / MD / TXT / DOCX）
- [x] 文本清洗（去除 Word 文档中的 LaTeX 标记）
- [x] 混合检索（BM25 + 向量）
- [x] CrossEncoder Rerank 精排
- [x] 多轮对话查询改写
- [x] 拒答机制
- [x] 引用来源溯源
- [x] 流式输出
- [x] RAGAS 量化评估
- [x] 消融实验
- [x] 拒答专项测试

### 待优化

- [ ] 支持多文档知识库（一次上传多份文档）
- [ ] 本地 LLM 部署（Ollama + Qwen，数据不出内网）
- [ ] 基于相似度阈值的拒答机制（而非 Prompt 兜底）
- [ ] 云端部署（Streamlit Cloud / HuggingFace Spaces）
- [ ] 支持表格/图片文档的 OCR 预处理
- [ ] 交叉裁判评估（引入 GPT-4o 或 Qwen）

## 📝 开发日志

- **Phase 1 - 基础链路**：环境搭建、文档加载、切片、向量化、Chroma 存储
- **Phase 2 - 检索优化**：混合检索（BM25 + 向量）、CrossEncoder Rerank 精排、拒答机制
- **Phase 3 - 前端交互**：Streamlit 界面、多轮对话查询改写、流式输出
- **Phase 4 - 量化评估**：RAGAS 四指标评估、Rerank 消融实验、拒答专项测试
- **Phase 5 - 工程整理**：代码重构（rag_core 模块化）、README 编写、GitHub 发布

## 🐛 问题与应对

### 1. Chroma 换文档后串库

**现象**：上传文档 A 并提问后，切换到文档 B，检索结果里仍然混着 A 的片段。

**原因**：`Chroma.from_documents()` 默认使用固定的 collection 名 `langchain`。同一进程内多次调用时，Chroma 会复用已有 collection，导致新旧文档的向量混在一起。

**解决**：每次构建向量库时生成唯一的 `collection_name`（基于 uuid），物理隔离不同文档的向量。

### 2. Word 文档中的 LaTeX 标记干扰检索

**现象**：Word 导出的文档中百分数写成 `\\(90\\%\\)`，用户搜"90%"匹配不上，BM25 命中率低。

**原因**：`Docx2txtLoader` 提取的是字面字符串，不会自动转换 LaTeX 标记。

**解决**：切片后增加文本清洗步骤，用正则把 `\\(...\\)` 转成 `...`、`\\%` 转成 `%`。修复后 Context Precision 从 0.44 提升到 0.65。

### 3. 大文档下初检召回不足

**现象**：在 37 页文档（134 个片段）上，部分问题直接触发拒答，但文档里明明有答案。

**原因**：初检召回数 `top_k=6` 太小，正确答案没进候选集，Rerank 也无法补救。

**解决**：把 `top_k` 从 6 提高到 15，召回率显著改善。这也说明 Rerank 的价值依赖于"初检召回足够多的候选"。

### 4. 扫描件 PDF 与旧版 .doc 无法处理

**现象**：上传扫描版 PDF 后系统"答非所问"，上传 `.doc` 直接报错。

**原因**：扫描件没有文字层，`PyPDFLoader` 提取为空；`.doc` 是二进制格式，`Docx2txtLoader` 不支持。

**解决**：加载后检查文字总量，少于 50 字符时主动报错并提示用户做 OCR；对 `.doc` 后缀给出明确的转换提示，避免用户困惑。

### 5. RAGAS 与 LangChain 1.x 的兼容性问题

**现象**：`from ragas import evaluate` 直接报 `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`。

**原因**：RAGAS 0.4.3 硬编码导入了 `langchain-community` 0.4.2 中已移除的 `vertexai` 子模块。

**解决**：在 import ragas 之前，向 `sys.modules` 注册一个空的 `vertexai` 模块作为兼容垫片。同时注意到 RAGAS 的 `answer_relevancy` 默认请求 `n=3`，而 DeepSeek 只支持 `n=1`，通过设置 `strictness = 1` 修复。

### 6. RAGAS 评估的同源偏见

**问题**：用 DeepSeek 生成答案，又用 DeepSeek 当裁判，存在"自己批改自己"的偏见。

**处理**：RAGAS 通过拆解任务（文本蕴含判断、反向生成问题）降低主观性，但无法完全消除。在 README 中主动说明这一局限，并指出未来可引入 GPT-4o 或 Qwen 作为独立裁判交叉验证。

## 📄 License

MIT

---

如果这个项目对你有帮助，欢迎 Star ⭐