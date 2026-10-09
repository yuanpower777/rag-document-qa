"""离线自测：不依赖大模型接口，只验证「切分 + 向量化 + 检索」是否有效。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag  # noqa: E402

doc = rag.load_text("data/sample.txt")
chunks = rag.split_text(doc)
print(f"文档字符数: {len(doc)}")
print(f"切分块数: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"  chunk[{i}] len={len(c)} :: {c[:40]!r}")

store = rag.VectorStore()
n = store.add_document(doc, "sample.txt")
print(f"已入库块数: {n}, 矩阵形状: {store.matrix.shape}")

cases = [
    ("退货政策是什么？", "7 天"),
    ("金卡会员有什么权益？", "金卡"),
    ("积分会过期吗？", "12"),
    ("发票怎么开？", "发票"),
    ("多久能送到？", "配送"),
    ("客服电话多少？", "400"),
]

hits_ok = 0
for query, expect in cases:
    hits = store.search(query, top_k=3)
    joined = " ".join(h.text for h in hits)
    ok = expect in joined
    hits_ok += ok
    print(f"\nQ: {query}  (期望包含“{expect}”) -> {'命中' if ok else '未命中'}")
    for rank, h in enumerate(hits, start=1):
        print(f"   [{rank}] score={h.score:.3f} {h.text[:60].replace(chr(10), ' ')}...")

print(f"\nTop-3 召回命中率: {hits_ok}/{len(cases)} = {hits_ok / len(cases):.0%}")

print("\n--- Prompt 预览 ---")
print(rag.build_prompt("金卡会员有什么权益？", store.search("金卡会员有什么权益？", 2))[:600])
