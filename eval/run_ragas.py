"""阶段6：用 Ragas 评估 RAG 质量（无需人工标注答案的指标）。

- faithfulness：答案是否忠于检索到的资料（防编造，最关键）。
- answer_relevancy：答案是否切题。
- 负样本（资料未提及）单独核对是否守住"不编造"。

testset.jsonl 每行 {question, note}。本脚本对每题跑检索+生成，喂给 Ragas。
评判用的 LLM/embedding 复用 config 里的 llm/embedding（OpenAI 兼容）。

    python -m eval.run_ragas
    python -m eval.run_ragas --limit 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.common import load_config, require_env
from rag.prompts import build_messages
from rag.query import retrieve

ROOT = Path(__file__).resolve().parent


def gen(question: str, cfg: dict):
    from openai import OpenAI

    contexts = retrieve(question, cfg)
    lc = cfg["llm"]
    client = OpenAI(base_url=lc["base_url"], api_key=require_env(lc["api_key_env"]))
    reply = client.chat.completions.create(
        model=lc["model"], messages=build_messages(question, contexts),
        temperature=lc["temperature"],
    ).choices[0].message.content
    return reply, [c.text for c in contexts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()

    rows = [json.loads(l) for l in (ROOT / "testset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    samples = {"question": [], "answer": [], "contexts": []}
    for r in rows:
        ans, ctxs = gen(r["question"], cfg)
        samples["question"].append(r["question"])
        samples["answer"].append(ans)
        samples["contexts"].append(ctxs)
        print(f"\nQ: {r['question']}\nA: {ans[:200]}...")

    # --- Ragas 评估 ---
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    lc, ec = cfg["llm"], cfg["embedding"]
    judge = ChatOpenAI(model=lc["model"], base_url=lc["base_url"],
                       api_key=require_env(lc["api_key_env"]), temperature=0)
    emb = OpenAIEmbeddings(model=ec["model"], base_url=ec["base_url"],
                           api_key=require_env(ec["api_key_env"]))
    result = evaluate(
        Dataset.from_dict(samples),
        metrics=[faithfulness, answer_relevancy],
        llm=judge, embeddings=emb,
    )
    print("\n=== Ragas 结果 ===")
    print(result)


if __name__ == "__main__":
    main()
