"""
RAGAS 评估脚本

用法：
    python eval_rag.py                # 加 Rerank（默认）
    python eval_rag.py --no-rerank    # 不加 Rerank（消融实验）
"""
import json
import argparse

from dotenv import load_dotenv

load_dotenv()

# ★ 必须先 import rag_core（内部含 RAGAS 兼容垫片），再 import ragas
from rag_core import (
    RAGSystem,
    load_local_document,
    get_embeddings,
    get_llm,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

answer_relevancy.strictness = 1

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def main():
    parser = argparse.ArgumentParser(description="RAGAS 评估")
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="关闭 Rerank（用于消融实验）",
    )
    parser.add_argument(
        "--doc", default="bp.docx", help="待评估文档路径",
    )
    parser.add_argument(
        "--questions", default="eval_questions.json",
        help="测试集 JSON 路径",
    )
    args = parser.parse_args()

    use_rerank = not args.no_rerank
    output_file = (
        "eval_result_with_rerank.csv" if use_rerank
        else "eval_result_no_rerank.csv"
    )
    mode_label = "加 Rerank" if use_rerank else "不加 Rerank"

    print(f"=== 评估模式：{mode_label} ===")
    print(f"=== 文档：{args.doc} ===\n")

    # 1. 构建 RAG 系统
    docs = load_local_document(args.doc)
    rag = RAGSystem(
        docs,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        top_k=DEFAULT_TOP_K,
    )
    print(f"切片数：{len(rag.chunks)}\n")

    # 2. 加载测试集
    with open(args.questions, "r", encoding="utf-8") as f:
        test_items = json.load(f)

    # 3. 逐条运行 RAG
    questions, answers, contexts, ground_truths = [], [], [], []
    for i, item in enumerate(test_items, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"[{i}/{len(test_items)}] {q}")

        answer, top_docs, _ = rag.answer(q, use_rerank=use_rerank)
        ctx = [d.page_content for d in top_docs]

        print(f"  → 答案：{answer[:100]}")
        print(f"  → 检索到 {len(ctx)} 个片段")

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)

    # 4. RAGAS 评估
    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    })
    evaluator_llm = LangchainLLMWrapper(get_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    print("\n开始 RAGAS 评估，可能需要几分钟...\n")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness, answer_relevancy,
            context_precision, context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print("\n===== 总体得分 =====")
    print(result)

    df = result.to_pandas()
    print("\n===== 逐条明细 =====")
    print(df.to_string(index=False))

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到 {output_file}")


if __name__ == "__main__":
    main()