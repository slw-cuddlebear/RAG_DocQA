"""
LLM 辅助动态切片模块

流程：
1. 从文档中提取段落
2. 编号并分批
3. 调用 LLM 判断切分点
4. 校验 + 兜底
5. 构建最终切片
"""
import re
from langchain_core.documents import Document

from rag_core import get_llm

# ========== 参数 ==========
BATCH_SIZE = 150  # 每批处理的段落数
BATCH_OVERLAP = 10  # 相邻批次重叠的段落数（保证边界一致）
MIN_CHUNK_CHARS = 80  # 切片最短字符数
MAX_CHUNK_CHARS = 800  # 切片最长字符数


# ========== 第一步：提取段落 ==========
def extract_paragraphs(text):
    """从文档文本中提取段落，过滤掉过短的"""
    # 按连续换行切分（保留单换行内的内容）
    raw = re.split(r'\n\s*\n', text)
    paragraphs = []
    for p in raw:
        p = p.strip()
        # 过滤掉空行、纯符号行、过短的
        if len(p) < 10:
            continue
        # 过滤掉纯页码、纯空白
        if re.fullmatch(r'[\d\s\.\-]+', p):
            continue
        paragraphs.append(p)
    return paragraphs


# ========== 第二步：分批 ==========
def batch_paragraphs(paragraphs, batch_size=BATCH_SIZE, overlap=BATCH_OVERLAP):
    """把段落分批，相邻批次有重叠"""
    batches = []
    i = 0
    while i < len(paragraphs):
        batch = paragraphs[i:i + batch_size]
        batches.append((i, batch))  # (起始索引, 段落列表)
        i += batch_size - overlap
    return batches


# ========== 第三步：LLM 判断切分点 ==========
SPLIT_PROMPT = """你是文档切分助手。下面是一份文档的段落列表，每个段落前面有编号。

请判断段落之间的语义边界，把内容相关的段落分为一组。每组应该围绕一个独立的主题或知识点。

规则：
1. 如果两个相邻段落讲的是同一件事，不要切开。
2. 如果主题发生变化（例如从"项目背景"转到"产品功能"），在变化处切开。
3. 每组包含的段落数不要太多，也不要太少（一般 2~8 段）。
4. 只输出分组，不要任何解释、不要用 markdown 代码块。
5. 输出格式（每行一组，用连字符连接起止编号）：
   1-3
   4-6
   7-10

段落列表：
{paragraphs}

输出："""


def call_llm_for_batch(paragraphs, start_idx):
    """调用 LLM 判断一批段落的切分点，返回切分点列表"""
    # 构造带编号的段落文本
    numbered = "\n".join(
        f"[{start_idx + i + 1}] {p[:200]}"  # 每段截断 200 字，控制 token
        for i, p in enumerate(paragraphs)
    )

    prompt = SPLIT_PROMPT.format(paragraphs=numbered)
    response = get_llm().invoke(prompt).content.strip()

    # 解析 LLM 输出："1-3\n4-6\n7-10" → [(1,3), (4,6), (7,10)]
    groups = []
    for line in response.split("\n"):
        line = line.strip()
        match = re.match(r'(\d+)\s*-\s*(\d+)', line)
        if match:
            s, e = int(match.group(1)), int(match.group(2))
            groups.append((s, e))

    return groups


# ========== 第四步：校验 ==========
def validate_groups(groups, total_paragraphs, start_idx):
    """
    校验 LLM 返回的切分点：
    - 编号是否在范围内
    - 是否覆盖全部段落（无遗漏）
    - 是否连续（无重叠）
    """
    if not groups:
        return False, "没有解析到任何分组"

    # 转换为全局编号
    global_groups = [(s, e) for s, e in groups]

    # 检查范围
    for s, e in global_groups:
        if s < start_idx + 1 or e > start_idx + total_paragraphs:
            return False, f"编号 {s}-{e} 超出范围"
        if s > e:
            return False, f"编号 {s}-{e} 顺序错误"

    # 检查连续性和完整性
    sorted_groups = sorted(global_groups, key=lambda x: x[0])
    expected_start = start_idx + 1
    for s, e in sorted_groups:
        if s != expected_start:
            return False, f"编号不连续：期望 {expected_start}，实际 {s}"
        expected_start = e + 1

    return True, "ok"


# ========== 第五步：兜底修复 ==========
def fallback_split(paragraphs, start_idx, end_idx):
    """按段落规则切分（兜底方案）"""
    groups = []
    current_start = start_idx + 1
    current_len = 0

    for i, p in enumerate(paragraphs):
        para_idx = start_idx + i + 1
        current_len += len(p)

        # 达到最大长度就切
        if current_len >= MAX_CHUNK_CHARS // 2:
            groups.append((current_start, para_idx))
            current_start = para_idx + 1
            current_len = 0

    # 收尾
    if current_start <= end_idx:
        groups.append((current_start, end_idx))

    return groups


# ========== 主函数 ==========
def dynamic_split(docs, progress_callback=None):
    """
    LLM 辅助动态切片

    参数：
    - docs: Document 对象列表
    - progress_callback: 可选回调函数，接收 (percent: int, message: str)

    返回：
    - chunks: Document 对象列表（切片后）
    """

    def report(p, msg):
        if progress_callback:
            progress_callback(p, msg)

    # 1. 提取全部段落（5%）
    report(5, "正在提取文档段落...")
    all_paragraphs = []
    source_name = docs[0].metadata.get("source", "未知") if docs else "未知"

    for doc in docs:
        all_paragraphs.extend(extract_paragraphs(doc.page_content))

    if not all_paragraphs:
        report(100, "文档内容为空")
        return []

    # 2. 分批（15%）
    report(15, f"共 {len(all_paragraphs)} 个段落，准备分批处理...")
    batches = batch_paragraphs(all_paragraphs)
    total_batches = len(batches)

    # 3. LLM 分批处理（20%~80%）
    all_groups = []
    for batch_idx, (start_idx, batch) in enumerate(batches):
        pct = 20 + int(60 * (batch_idx + 1) / total_batches)
        report(pct, f"正在分析第 {batch_idx + 1}/{total_batches} 批段落...")

        try:
            groups = call_llm_for_batch(batch, start_idx)
            valid, reason = validate_groups(groups, len(batch), start_idx)

            if valid:
                # 去重（相邻批次有重叠）
                for s, e in groups:
                    if not all_groups or s > all_groups[-1][1]:
                        all_groups.append((s, e))
            else:
                # 校验失败 → 兜底
                print(f"[WARN] 批次 {batch_idx + 1} 校验失败：{reason}，使用兜底切分")
                all_groups.extend(
                    fallback_split(batch, start_idx, start_idx + len(batch) - 1)
                )
        except Exception as e:
            print(f"[ERROR] 批次 {batch_idx + 1} 调用失败：{e}，使用兜底切分")
            all_groups.extend(
                fallback_split(batch, start_idx, start_idx + len(batch) - 1)
            )

    # 4. 校验全局覆盖（85%）
    report(85, "正在校验切分结果...")
    all_groups = sorted(set(all_groups), key=lambda x: x[0])

    # 5. 构建切片（95%）
    report(95, "正在构建切片...")
    chunks = []
    for s, e in all_groups:
        # 段落编号从 1 开始，列表索引从 0 开始
        content = "\n\n".join(all_paragraphs[s - 1:e])

        # 长度兜底
        if len(content) > MAX_CHUNK_CHARS:
            # 超长就按 MAX_CHUNK_CHARS 再切
            for i in range(0, len(content), MAX_CHUNK_CHARS):
                chunks.append(Document(
                    page_content=content[i:i + MAX_CHUNK_CHARS],
                    metadata={"source": source_name},
                ))
        elif len(content) < MIN_CHUNK_CHARS:
            # 过短就合并到上一块
            if chunks:
                chunks[-1].page_content += "\n\n" + content
            else:
                chunks.append(Document(
                    page_content=content,
                    metadata={"source": source_name},
                ))
        else:
            chunks.append(Document(
                page_content=content,
                metadata={"source": source_name},
            ))

    report(100, f"完成，共生成 {len(chunks)} 个切片")
    return chunks