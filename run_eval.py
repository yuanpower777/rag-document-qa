"""批量检索测试：对题库跑一遍，统计召回命中率。

用法：
    python run_eval.py 你的报告.docx
    python run_eval.py 你的报告.docx --top-k 5 --chunk-size 500
    python run_eval.py 你的报告.docx --ask      # 额外调用大模型生成答案（需 API Key）

输出：每个问题的命中情况 + 汇总命中率，并写出 eval_result.json 供进一步分析。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import rag


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索效果批量评测")
    parser.add_argument("document", help="知识库文档（.docx/.pdf/.txt/.md）")
    parser.add_argument("--questions", default="test_questions.json")
    parser.add_argument("--top-k", type=int, default=rag.TOP_K)
    parser.add_argument("--chunk-size", type=int, default=rag.CHUNK_SIZE)
    parser.add_argument("--ask", action="store_true", help="额外让大模型生成答案")
    args = parser.parse_args()

    rag.CHUNK_SIZE = args.chunk_size

    with open(args.questions, "r", encoding="utf-8") as fh:
        suite = json.load(fh)
    questions = suite["questions"]

    text = rag.load_text(args.document)
    store = rag.VectorStore()
    n_chunks = store.add_document(text, os.path.basename(args.document))
    print(f"文档：{args.document}")
    print(f"字符数 {len(text)}　切分块数 {n_chunks}　chunk_size={args.chunk_size}　Top-K={args.top_k}")
    print("=" * 70)

    results = []
    hit_count = 0
    judged = 0          # 只对 A/B/C 类（有期望关键词）统计召回
    for item in questions:
        keywords = item.get("expected_keywords") or []
        hits = store.search(item["question"], top_k=args.top_k)
        joined = " ".join(h.text for h in hits)
        matched = [k for k in keywords if k in joined]

        if keywords:
            judged += 1
            hit = len(matched) == len(keywords)
            hit_count += hit
            status = "命中" if hit else f"缺 {'/'.join(set(keywords) - set(matched))}"
        else:
            hit = None
            status = "兜底题（无期望关键词，需人工看是否编造）"

        print(f"[{item['id']:>2}] {item['category']:<10} {item['question']}")
        print(f"     -> {status}   (Top-1 score={hits[0].score:.3f})" if hits else "     -> 无召回")

        record = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_keywords": keywords,
            "matched_keywords": matched,
            "hit": hit,
            "top_hits": [
                {"score": round(h.score, 4), "source": h.source, "text": h.text[:400]}
                for h in hits
            ],
        }
        if args.ask:
            try:
                record["answer"] = rag.generate(item["question"], hits)
            except Exception as exc:  # noqa: BLE001
                record["answer"] = f"生成失败：{exc}"
        results.append(record)

    print("=" * 70)
    if judged:
        rate = hit_count / judged
        print(f"检索召回命中率：{hit_count}/{judged} = {rate:.0%}   "
              f"(chunk_size={args.chunk_size}, Top-K={args.top_k})")
        print("说明：命中率按「该题期望关键词是否全部出现在 Top-K 片段中」判定，")
        print("      这是检索层的粗略指标；答案质量需再人工核对 answer_points。")

    with open("eval_result.json", "w", encoding="utf-8") as fh:
        json.dump({
            "document": args.document,
            "chunk_size": args.chunk_size,
            "top_k": args.top_k,
            "chunks": n_chunks,
            "hit_rate": (hit_count / judged) if judged else None,
            "results": results,
        }, fh, ensure_ascii=False, indent=2)
    print("详细结果已写入 eval_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
