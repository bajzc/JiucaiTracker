"""阶段4：把 chunk 向量化后写入 Qdrant。

- 向量：BGE-M3 dense（COSINE）。
- payload：保留 video_id/title/day_index/时间戳/text/stocks，
  并对 stocks、day_index 建索引，支持"问单只股票"时按 stocks 过滤、
  按 day_index 做时间范围过滤。
- 召回策略：dense 召回 + query.py 里的 bge-reranker 精排。
  （若要 sparse 真混检，可后续接 FastEmbed BM25，结构已预留。）

幂等：point id 由 chunk_id 派生（uuid5），重复跑覆盖而不重复。

用法:
    python -m pipeline.index                 # 读取 data/chunks/*.json 建库
    python -m pipeline.index --recreate      # 删除并重建 collection
"""
from __future__ import annotations

import argparse
import json
import uuid

from tqdm import tqdm

from .common import load_config, resolve
from .embedding import embed

NAMESPACE = uuid.UUID("0d6f3c2e-9b1a-4c7e-8f00-1a2b3c4d5e6f")


def load_chunks(cfg: dict) -> list[dict]:
    cdir = resolve(cfg["paths"]["chunks"])
    chunks: list[dict] = []
    for f in sorted(cdir.glob("*.json")):
        chunks.extend(json.loads(f.read_text(encoding="utf-8")))
    return chunks


def get_client(cfg: dict):
    from qdrant_client import QdrantClient

    return QdrantClient(url=cfg["vector_store"]["url"])


def ensure_collection(client, cfg: dict, recreate: bool) -> None:
    from qdrant_client import models

    name = cfg["vector_store"]["collection"]
    dim = cfg["embedding"]["dim"]
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        # 便于按个股 / 交易日过滤
        client.create_payload_index(name, "stocks", models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "day_index", models.PayloadSchemaType.INTEGER)
        client.create_payload_index(name, "video_id", models.PayloadSchemaType.KEYWORD)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    from qdrant_client import models

    cfg = load_config()
    chunks = load_chunks(cfg)
    if not chunks:
        raise SystemExit("没有 chunk，请先跑 build_chunks")

    client = get_client(cfg)
    ensure_collection(client, cfg, args.recreate)
    name = cfg["vector_store"]["collection"]

    for i in tqdm(range(0, len(chunks), args.batch), desc="建库"):
        batch = chunks[i : i + args.batch]
        vectors = embed([c["text"] for c in batch])
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(NAMESPACE, c["chunk_id"])),
                vector=vec,
                payload=c,
            )
            for c, vec in zip(batch, vectors)
        ]
        client.upsert(name, points=points)

    info = client.get_collection(name)
    print(f"已写入 {len(chunks)} 个 chunk，collection={name} 现有 {info.points_count} 点")


if __name__ == "__main__":
    main()
