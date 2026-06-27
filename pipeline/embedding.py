"""共享：BGE-M3 向量化（走 OpenAI 兼容的 /embeddings，默认 SiliconFlow 托管）。

index.py 建库与 query.py 检索都用它，保证查询/文档同模型同维度。
"""
from __future__ import annotations

from functools import lru_cache

from .common import dbg, is_debug, load_config, require_env


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    cfg = load_config()["embedding"]
    return OpenAI(base_url=cfg["base_url"], api_key=require_env(cfg["api_key_env"]))


def embed(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    cfg = load_config()["embedding"]
    batch_size = batch_size or int(cfg.get("batch_size", 10))
    client = _client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if is_debug():
            dbg(f"EMBED · {cfg['model']}", f"本批 {len(batch)} 条；首条: {batch[0][:80]}…")
        resp = client.embeddings.create(model=cfg["model"], input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
