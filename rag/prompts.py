"""阶段5（提示词）：强约束"基于资料、可溯源、不编造"。"""
from __future__ import annotations

from .query import Context

SYSTEM = """你是一个炒股视频内容的问答助手。严格遵守：
1. 只依据【资料】里的内容回答，不得引入资料之外的知识或编造个股、价位、数字。
2. 每个关键结论后用 [序号] 标注出处（对应资料编号）。
3. 若资料中没有相关信息，直接回答“资料中未提及”，不要猜测。
4. 这些是 UP 主在某一天的实盘观点，带有时效性。回答末尾用一句话提醒：
   “以上为对应日期的个人观点，非投资建议，注意时效。”
5. 当被问某只股票的买卖逻辑时，归纳 UP 主的理由（基本面/技术面/仓位），
   并指出是哪一天、什么背景下的判断。"""


def build_messages(
    question: str,
    contexts: list[Context],
    history: list[dict] | None = None,
) -> list[dict]:
    if not contexts:
        blocks = "（无检索结果）"
    else:
        blocks = "\n\n".join(
            f"[{i + 1}] 出处：{c.cite()}\n{c.text}" for i, c in enumerate(contexts)
        )
    user = f"【资料】\n{blocks}\n\n【问题】\n{question}"
    return [
        {"role": "system", "content": SYSTEM},
        *(history or []),
        {"role": "user", "content": user},
    ]
