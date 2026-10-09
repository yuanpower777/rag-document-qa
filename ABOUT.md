# 一、项目简介

**智能文档问答助手（RAG）** 是一个基于检索增强生成的文档问答应用。

把大模型直接当文档问答用会遇到三个问题：它不知道你的私有文档、不知道时会编造内容（幻觉）、
答案无法溯源。本项目通过 RAG 解决：先从你的文档中检索出相关片段，再让大模型**只依据这些片段**
作答，并强制标注引用编号，使答案有据可查、可追溯到原文。

支持 PDF / Word / Markdown / TXT 文档，提供命令行与网页两种交互方式，核心检索逻辑不依赖
FAISS / PyTorch 等重型库，普通笔记本即可跑通。

## 主要特性

- 多格式文档加载（`.txt` / `.md` / `.pdf` / `.docx`，Word 正文与表格均可解析）
- 按自然边界切分的文本分块（可配置粒度与重叠），避免切断句子
- 两种向量化后端：本地零依赖兜底方案，或接入真实语义 embedding 接口
- 基于余弦相似度的 Top-K 向量检索（numpy 实现，无需编译扩展）
- 受约束的答案生成：强制"只依据资料回答 + 标注引用编号"，`temperature=0.2` 降低随机性
- 答案溯源：网页界面可展开查看每条回答引用的原文片段与相似度
- Streamlit 网页界面：多文件上传建索引、参数实时调节、流式输出、多轮对话
- 批量评测脚本：可对整套测试题统计召回命中率，用于对比不同参数

## 技术栈

Python 3.10+ · numpy · Streamlit · OpenAI 兼容大模型 API（默认 DeepSeek）· pypdf · python-docx

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

# 配置 API Key（PowerShell 示例）
$env:DEEPSEEK_API_KEY="sk-xxxxxxxx"

# 命令行问答
python cli.py data/sample.txt -q "金卡会员有什么权益？"

# 网页界面
streamlit run app.py

# 离线自测（无需 API Key、无需联网）
python selftest.py
```

## 评测结果

以仓库内置的实验报告为语料（约 7200 字）、20 道测试题为基准，
对比不同切分粒度与召回数量的检索命中率：

| chunk_size | 切分块数 | Top-K=3 | Top-K=4 | Top-K=6 |
|---|---|---|---|---|
| 300 | 39 | 44% | 56% | 67% |
| 500 | 19 | 61% | 78% | 83% |
| 800 | 12 | 61% | 78% | 94% |

命中率随粒度与召回数单调上升，但并非越大越好：块越大越容易"顺带包含"关键词导致指标虚高，
且会把大量冗余文本塞进 Prompt，更慢更贵。默认取 `chunk_size=500, Top-K=4` 作为平衡点。
详见 README「五、输入输出示例」。

## 项目结构

```
├── app.py                 Streamlit 网页界面
├── cli.py                 命令行问答入口
├── rag.py                 核心：加载 / 切分 / 向量化 / 检索 / 生成
├── run_eval.py            批量检索评测脚本
├── selftest.py            零依赖离线自测
├── test_questions.json    20 道测试题
├── requirements.txt
├── LICENSE
└── data/
    ├── sample.txt
    └── 机器学习实验报告-评测语料.txt
```
