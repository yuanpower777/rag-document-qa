"""核心 RAG 流程：文档加载、切分、向量化、检索、生成。

设计原则：默认零编译依赖（不依赖 faiss / torch），用 numpy 做向量检索，
方便在普通笔记本上直接跑通，同时保留切换到真实 embedding 接口的能力。
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# ----------------------------------------------------------------------------
# 配置：通过环境变量覆盖，方便对比实验
# ----------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "4"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "")            # 留空则用本地兜底向量化
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or LLM_API_KEY

SYSTEM_PROMPT = (
    "你是一个严谨的文档问答助手。只能依据【参考资料】中的内容回答问题。"
    "如果参考资料中没有足够信息，必须直接回答“资料中未提及”，不要编造。"
    "回答要条理清晰，并在句末用 [1] [2] 这样的编号标注依据来源。"
)


# ----------------------------------------------------------------------------
# 文档加载与切分
# ----------------------------------------------------------------------------
def load_text(path: str) -> str:
    """读取 .txt/.md/.pdf/.docx 文档为纯文本。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("解析 PDF 需要 pypdf：pip install pypdf") from exc
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        return load_docx(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def load_docx(path: str) -> str:
    """读取 Word 文档，按文档顺序拼接正文段落与表格内容。"""
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("解析 DOCX 需要 python-docx：pip install python-docx") from exc

    document = Document(path)
    blocks: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text = Paragraph(child, document).text.strip()
            if text:
                blocks.append(text)
        elif tag == "tbl":
            table = Table(child, document)
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                # 合并单元格的文本会重复出现，去重后保留顺序
                deduped = list(dict.fromkeys(cells))
                line = " | ".join(c for c in deduped if c)
                if line:
                    blocks.append(line)
    return "\n".join(blocks)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int | None = None,
               overlap: int | None = None) -> list[str]:
    """按字符窗口切分，优先在句末/换行处断开，减少语义截断。

    注意：chunk_size/overlap 默认值在函数体内从模块变量读取，
    这样运行时修改 rag.CHUNK_SIZE 才能真正生效（避免默认参数在导入时绑定）。
    """
    chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = CHUNK_OVERLAP if overlap is None else overlap
    text = clean_text(text)
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            # 在窗口后半段寻找更自然的断点
            for sep in ("\n\n", "\n", "。", "；", ". ", "！", "？"):
                pos = window.rfind(sep)
                if pos != -1 and pos > chunk_size * 0.5:
                    end = start + pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# ----------------------------------------------------------------------------
# 向量化：本地兜底（hashing trick）/ 真实 embedding 接口
# ----------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    """中英混合的粗粒度分词：英文按词，中文按字 + 二元组。"""
    raw = _TOKEN_RE.findall(text.lower())
    grams = list(raw)
    cjk = [t for t in raw if "\u4e00" <= t <= "\u9fff"]
    grams += [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return grams


def local_embed(texts: Sequence[str], dim: int | None = None) -> np.ndarray:
    """零依赖的向量化兜底方案：hashing + 二元组特征，再做 L2 归一化。

    注意：这不是语义模型，检索能力弱于真实 embedding，但足以完整跑通
    RAG 流程，也便于在无外网/无额度的环境下做工程验证。
    """
    dim = EMBED_DIM if dim is None else dim
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in _tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vectors[row, idx] += sign
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def api_embed(texts: Sequence[str]) -> np.ndarray:
    """调用 OpenAI 兼容的 embedding 接口。"""
    from openai import OpenAI

    client = OpenAI(api_key=EMBED_API_KEY, base_url=LLM_BASE_URL)
    resp = client.embeddings.create(model=EMBED_MODEL, input=list(texts))
    vectors = np.array([item.embedding for item in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed(texts: Sequence[str]) -> np.ndarray:
    if EMBED_MODEL:
        return api_embed(texts)
    return local_embed(texts)


# ----------------------------------------------------------------------------
# 向量库
# ----------------------------------------------------------------------------
@dataclass
class Hit:
    text: str
    score: float
    source: str
    chunk_id: int


class VectorStore:
    """极简向量索引：内存矩阵 + 余弦相似度（向量已归一化，点积即余弦）。"""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.matrix: np.ndarray | None = None

    def add_document(self, text: str, source: str) -> int:
        parts = split_text(text)
        if not parts:
            return 0
        vectors = embed(parts)
        self.chunks.extend(parts)
        self.sources.extend([source] * len(parts))
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])
        return len(parts)

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = TOP_K if top_k is None else top_k
        if self.matrix is None or not self.chunks:
            return []
        qvec = embed([query])[0]
        scores = self.matrix @ qvec
        order = np.argsort(-scores)[:top_k]
        return [
            Hit(text=self.chunks[i], score=float(scores[i]),
                source=self.sources[i], chunk_id=int(i))
            for i in order
        ]


# ----------------------------------------------------------------------------
# Prompt 组装与生成
# ----------------------------------------------------------------------------
def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] 来源：{hit.source}（相似度 {hit.score:.3f}）\n{hit.text}")
    context = "\n\n".join(blocks) if blocks else "（未检索到相关内容）"
    return (
        f"【参考资料】\n{context}\n\n"
        f"【用户问题】\n{question}\n\n"
        "请依据上述参考资料回答，并在引用处标注 [编号]。"
    )


def generate(question: str, hits: Sequence[Hit], history: Iterable[dict] | None = None,
             stream: bool = False):
    """调用大模型生成回答。stream=True 时返回逐段文本的生成器。"""
    if not LLM_API_KEY:
        raise RuntimeError("未配置 API Key，请设置环境变量 DEEPSEEK_API_KEY")
    from openai import OpenAI

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        messages.append(turn)
    messages.append({"role": "user", "content": build_prompt(question, hits)})

    if stream:
        def iterator():
            resp = client.chat.completions.create(
                model=LLM_MODEL, messages=messages, temperature=0.2, stream=True)
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return iterator()

    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=0.2)
    return resp.choices[0].message.content or ""


def answer(store: VectorStore, question: str,
           history: Iterable[dict] | None = None) -> tuple[str, list[Hit]]:
    hits = store.search(question)
    return generate(question, hits, history), hits
