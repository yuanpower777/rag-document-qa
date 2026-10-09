"""命令行问答入口：便于快速验证与截图（无需启动网页）。

用法：
    python cli.py data/sample.txt
    python cli.py data/sample.txt -q "退货政策是什么？"
"""
from __future__ import annotations

import argparse
import sys

import rag


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 文档问答（命令行版）")
    parser.add_argument("docs", nargs="+", help="知识库文档路径（.txt/.md/.pdf/.docx）")
    parser.add_argument("-q", "--question", help="直接提问一次后退出")
    parser.add_argument("-k", "--top-k", type=int, default=rag.TOP_K)
    args = parser.parse_args()

    store = rag.VectorStore()
    for path in args.docs:
        text = rag.load_text(path)
        count = store.add_document(text, path)
        print(f"[索引] {path} -> {count} 个文本块")

    def ask(question: str) -> None:
        hits = store.search(question, top_k=args.top_k)
        print("\n--- 检索到的片段 ---")
        for rank, hit in enumerate(hits, start=1):
            preview = hit.text.replace("\n", " ")[:80]
            print(f"[{rank}] {hit.source} score={hit.score:.3f} :: {preview}...")
        print("\n--- 回答 ---")
        try:
            print(rag.generate(question, hits))
        except Exception as exc:  # noqa: BLE001
            print(f"生成失败：{exc}")

    if args.question:
        ask(args.question)
        return 0

    print("\n输入问题后回车（输入 exit 退出）：")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        ask(question)
    return 0


if __name__ == "__main__":
    sys.exit(main())
