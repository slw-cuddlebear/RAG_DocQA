"""
拒答能力评估脚本

用法：
    python eval_reject.py
"""
import json
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from rag_core import (
    RAGSystem,
    load_local_document,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_RERANK_TOP_K,
)


REJECT_KEYWORDS = [
    "无法回答", "没有相关", "资料中没有",
    "没有提到", "未提供", "无法确定", "没有明确",
]


def is_rejected(answer: str) -> bool:
    return any(k in answer for k in REJECT_KEYWORDS)


def main():
    # 1. 构建 RAG 系统
    docs = load_local_document("bp.docx")
    rag = RAGSystem(
        docs,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        top_k=DEFAULT_TOP_K,
    )
    print(f"切片数：{len(rag.chunks)}\n")

    # 2. 加载拒答测试集
    with open("reject_questions.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # 3. 逐条测试
    results = []
    correct = 0

    for i, case in enumerate(test_cases, 1):
        q = case["question"]
        expected = case["should_reject"]

        answer, _, _ = rag.answer(
            q, use_rerank=True, rerank_top_k=DEFAULT_RERANK_TOP_K
        )
        actual = is_rejected(answer)
        hit = (actual == expected)
        correct += hit
        status = "✅" if hit else "❌"

        results.append({
            "question": q,
            "category": case["category"],
            "expected_reject": expected,
            "actual_reject": actual,
            "hit": hit,
            "answer": answer[:120],
        })

        print(f"{status} [{i}/{len(test_cases)}] {q}")
        print(f"     预期拒答={expected}, 实际拒答={actual}")
        print(f"     答案：{answer[:80]}\n")

    # 4. 汇总
    total = len(test_cases)
    acc = correct / total

    print("=" * 60)
    print(f"拒答准确率：{correct}/{total} = {acc:.2%}")

    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_cat[r["category"]]["total"] += 1
        if r["hit"]:
            by_cat[r["category"]]["correct"] += 1

    print("\n各类别表现：")
    for cat, stat in by_cat.items():
        print(f"  {cat}: {stat['correct']}/{stat['total']}")

    pd.DataFrame(results).to_csv(
        "eval_reject_result.csv", index=False, encoding="utf-8-sig"
    )
    print("\n结果已保存到 eval_reject_result.csv")


if __name__ == "__main__":
    main()