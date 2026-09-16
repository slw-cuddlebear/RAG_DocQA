"""
提问模式核心模块：出题、解析、校验

设计原则：
- 只出简答题（避免判分逻辑）
- 一个片段出 1~3 道题，由 LLM 判断
- 三层防幻觉：Prompt 约束 + 格式校验 + 依据溯源
"""
import re

from rag_core import get_llm


# ========== 出题 Prompt ==========
QUIZ_PROMPT = """你是一个学习助手。请根据下面的文档片段，出 1~3 道简答题，帮助学生复习与记忆。

规则：
1. 题目必须完全基于文档内容，不得编造。
2. 优先考查关键概念、事实、原理、数据。
3. 每道题考查不同的知识点，避免重复。
4. 题目应清晰明确，答案应简洁完整。
5. 每道题必须给出"答案依据"，即答案对应的原文片段（直接引用原文，不超过 150 字）。
6. 如果这段内容信息量较大，可以出 2~3 道；如果信息量小，出 1 道即可。
7. 如果这段内容完全不适合出题（如封面、目录、页码、图表标题），
   请只输出一行："不适合出题"。不要尝试编题。

文档片段：
{context}

请严格按以下格式输出（多道题之间用 --- 分隔）：

【题目】...
【参考答案】...
【答案依据】...

---

【题目】...
【参考答案】...
【答案依据】...
"""


# ========== 解析单道题 ==========
def parse_single_question(block: str) -> dict | None:
    """解析单道题的三个字段，返回 dict 或 None"""
    q_match = re.search(r'【题目】(.*?)(?=【参考答案】|$)', block, re.DOTALL)
    a_match = re.search(r'【参考答案】(.*?)(?=【答案依据】|$)', block, re.DOTALL)
    s_match = re.search(r'【答案依据】(.*?)$', block, re.DOTALL)

    if not (q_match and a_match and s_match):
        return None

    question = q_match.group(1).strip()
    reference_answer = a_match.group(1).strip()
    reference_source = s_match.group(1).strip()

    if not question or not reference_answer or not reference_source:
        return None

    return {
        "question": question,
        "reference_answer": reference_answer,
        "reference_source": reference_source,
    }


# ========== 校验依据是否来自原文 ==========
def is_grounded(reference_source: str, chunk_text: str, threshold: float = 0.7) -> bool:
    """
    检查"答案依据"是否真的来自原文。
    用中文字符重叠率近似判断：依据中的中文字符有 70% 以上出现在原文 → 判定为可信。
    """
    if not reference_source:
        return False

    ref_chars = set(re.findall(r'[\u4e00-\u9fa5]', reference_source))
    src_chars = set(re.findall(r'[\u4e00-\u9fa5]', chunk_text))

    if not ref_chars:
        return False

    overlap = len(ref_chars & src_chars) / len(ref_chars)
    return overlap >= threshold


# ========== 主函数：从一个片段出题 ==========
def generate_questions(chunk_text: str, max_questions: int = 3) -> list[dict]:
    """
    从一个文档片段生成 1~3 道题。

    返回：
    - list[dict]：题目列表（可能为空）
      - 空列表含义：不适合出题，或全部题目校验失败
    """
    prompt = QUIZ_PROMPT.format(context=chunk_text)

    try:
        response = get_llm().invoke(prompt).content
    except Exception as e:
        print(f"[ERROR] 出题调用失败：{e}")
        return []

    # LLM 判定不适合出题
    if "不适合出题" in response:
        return []

    # 按 --- 切分多道题
    blocks = re.split(r'\n-{3,}\n', response)

    questions = []
    for block in blocks:
        q = parse_single_question(block)
        if q is None:
            continue

        # 校验答案依据是否来自原文
        if not is_grounded(q["reference_source"], chunk_text):
            print(f"[WARN] 丢弃依据不可信的题目：{q['question'][:40]}...")
            continue

        questions.append(q)

    return questions[:max_questions]


# ========== 测试入口 ==========
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # 用几种"不适合出题"的文本测试
    test_cases = [
        ("目录", "1.1 创意来源\t3\n1.2 项目目标\t4\n1.3 发展路径\t4"),
        ("封面", "AI智学课堂商业计划书\n智学助教——AI驱动的小学课堂智能练习与诊断系统"),
        ("页码", "页码 12\n页码 13"),
        ("正常内容", "本项目采用免费基础服务+增值付费服务的模式，免费版覆盖90%的课堂练习需求。"),
    ]

    for name, text in test_cases:
        print(f"\n===== 测试：{name} =====")
        results = generate_questions(text)
        if not results:
            print("✅ 正确拒答（无题目生成）")
        else:
            print(f"⚠️ 生成了 {len(results)} 道题：")
            for q in results:
                print(f"  - {q['question']}")