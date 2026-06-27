"""阶段5（检索）：dense 召回 → 个股 metadata 过滤 → bge-reranker 精排。

返回带出处的上下文片段，供生成端引用。
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from pipeline.build_chunks import extract_entities, load_lexicon
from pipeline.common import dbg, is_debug, load_config, require_env
from pipeline.embedding import embed_one


@dataclass
class Context:
    text: str
    title: str
    video_id: str
    day_index: int | None
    start_ts: float
    end_ts: float
    stocks: list[str]
    score: float

    def cite(self) -> str:
        m, s = divmod(int(self.start_ts), 60)
        day = f"第{self.day_index}日" if self.day_index is not None else "?"
        return f"《{self.title}》({day} @ {m:02d}:{s:02d})"


def _qdrant_client(cfg: dict):
    from qdrant_client import QdrantClient

    return QdrantClient(url=cfg["vector_store"]["url"])


def rewrite_query(question: str, history: list[dict], cfg: dict) -> str:
    """用对话历史将含指代词的问题改写为独立检索查询。历史为空时直接返回原问题。"""
    if not history:
        dbg("REWRITE · 查询改写", f"no history")
        return question
    import httpx as _httpx
    lc = cfg["llm"]
    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}" for m in history[-4:]
    )
    prompt = (
        "根据以下对话历史，将\"最新问题\"改写为一个独立、完整的检索查询，"
        "展开所有指代词（如\"这个\"\"它\"\"上面提到的\"等），保留关键实体。"
        "若问题本身已足够独立，原样返回。只输出改写后的查询，不要任何解释。\n\n"
        f"【对话历史】\n{history_text}\n\n"
        f"【最新问题】\n{question}\n\n"
        "【改写后的查询】"
    )
    headers = {"Authorization": f"Bearer {require_env(lc['api_key_env'])}",
               "Content-Type": "application/json"}
    payload = {
        "model": lc["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
    }
    url = lc["base_url"].rstrip("/") + "/chat/completions"
    resp = _httpx.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    rewritten = resp.json()["choices"][0]["message"]["content"].strip()
    dbg("REWRITE · 查询改写", f"原始: {question}\n改写: {rewritten}")
    return rewritten


def detect_stocks(question: str, cfg: dict) -> list[str]:
    lexicon = load_lexicon(cfg)
    return extract_entities(question, lexicon, cfg["entity"]["match_stock_code"])


def rerank(question: str, docs: list[str], cfg: dict, top_n: int) -> list[tuple[int, float]]:
    """返回 [(原文档下标, 相关度)]，按相关度降序。兼容阿里云/SiliconFlow 两种格式。"""
    rc = cfg["rerank"]
    headers = {"Authorization": f"Bearer {require_env(rc['api_key_env'])}"}
    if rc.get("format", "dashscope") == "dashscope":
        # 阿里云 gte-rerank：input.{query,documents} + parameters；结果在 output.results
        payload = {"model": rc["model"],
                   "input": {"query": question, "documents": docs},
                   "parameters": {"top_n": top_n, "return_documents": False}}
        results = _post(rc["base_url"], headers, payload)["output"]["results"]
    else:
        # SiliconFlow / OpenAI 兼容 rerank
        payload = {"model": rc["model"], "query": question, "documents": docs,
                   "top_n": top_n, "return_documents": False}
        results = _post(rc["base_url"], headers, payload)["results"]
    ranked = [(r["index"], r["relevance_score"]) for r in results]
    if is_debug():
        preview = "\n".join(f"  #{idx} score={score:.3f}  {docs[idx][:60]}…" for idx, score in ranked)
        dbg(f"RERANK · {rc['model']}  (取 top{top_n} / 候选 {len(docs)})",
            f"query: {question}\n{preview}")
    return ranked


def _post(url: str, headers: dict, payload: dict) -> dict:
    resp = httpx.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def retrieve(question: str, cfg: dict | None = None) -> list[Context]:
    from qdrant_client import models

    cfg = cfg or load_config()
    rcfg = cfg["retrieval"]
    client = _qdrant_client(cfg)

    # 问句若点名个股，则按 stocks 过滤，聚焦该股跨视频的讨论
    stocks = detect_stocks(question, cfg)
    qfilter = None
    if stocks:
        qfilter = models.Filter(should=[
            models.FieldCondition(key="stocks", match=models.MatchValue(value=s))
            for s in stocks
        ])
    if is_debug():
        dbg("RETRIEVE · 检索",
            f"问题: {question}\n命中个股(用于过滤): {stocks or '无 → 纯语义检索'}\n"
            f"top_k={rcfg['top_k']}  top_n={rcfg['top_n']}")

    hits = client.query_points(
        cfg["vector_store"]["collection"],
        query=embed_one(question),
        query_filter=qfilter,
        limit=rcfg["top_k"],
        with_payload=True,
    ).points
    # 命中过少（该股提及稀疏）时放宽：去掉过滤再召回一次
    if qfilter and len(hits) < rcfg["top_n"]:
        if is_debug():
            dbg("RETRIEVE · 放宽", f"过滤后仅 {len(hits)} 条 < top_n，去掉个股过滤重新召回")
        hits = client.query_points(
            cfg["vector_store"]["collection"],
            query=embed_one(question), limit=rcfg["top_k"], with_payload=True,
        ).points
    if is_debug():
        dbg("RETRIEVE · 召回结果", f"召回 {len(hits)} 块；标题预览:\n" +
            "\n".join(f"  - {h.payload['title'][:40]} @ {h.payload['start_ts']}s" for h in hits[:8]))

    if not hits:
        return []

    payloads = [h.payload for h in hits]
    ranked = rerank(question, [p["text"] for p in payloads], cfg, rcfg["top_n"])
    out = []
    for idx, score in ranked:
        p = payloads[idx]
        out.append(Context(
            text=p["text"], title=p["title"], video_id=p["video_id"],
            day_index=p.get("day_index"), start_ts=p["start_ts"], end_ts=p["end_ts"],
            stocks=p.get("stocks", []), score=score,
        ))
    return out
