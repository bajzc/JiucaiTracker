"""阶段3：清洗 + 实体抽取 + 分段。

把每个转写 JSON 的句段合并成 ~max_chars 的块，块间按句段回退实现重叠，
每块保留时间戳与命中的个股/板块（用于按股票过滤、跨视频聚合）。

实体抽取走"词典 + 股票代码正则"，确定、免费、可复现——比逐块调 LLM 更适合
建索引阶段。词典见 data/lexicon.txt。

用法:
    python -m pipeline.build_chunks
"""
from __future__ import annotations

import json
import re

from tqdm import tqdm

from .common import load_config, resolve

# 连续重复的语气词/口水（轻清洗，避免误删实质内容）
_FILLER = re.compile(r"(那个|这个|然后|就是说|嗯+|呃+|啊+)(?=\1)")
_WS = re.compile(r"\s+")
# A股6位代码 / 港股代码（00700、0700.HK 等）
_CODE = re.compile(r"\b(\d{6}|\d{4,5}\.?HK)\b", re.IGNORECASE)


def load_lexicon(cfg: dict) -> list[str]:
    path = resolve(cfg["entity"]["lexicon"])
    if not path.exists():
        return []
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    # 长词优先，避免短词先匹配（如"医药" vs "创新药"）
    return sorted(set(terms), key=len, reverse=True)


def clean(text: str) -> str:
    text = _FILLER.sub("", text)
    return _WS.sub("", text).strip()


def extract_entities(text: str, lexicon: list[str], match_code: bool) -> list[str]:
    hits = [term for term in lexicon if term in text]
    if match_code:
        hits += [m.group(1) for m in _CODE.finditer(text)]
    # 去重保序
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def chunk_segments(segments: list[dict], max_chars: int, overlap: int, min_chars: int) -> list[dict]:
    """合并句段成块；块间保留约 overlap 字的尾部句段做重叠。"""
    chunks: list[dict] = []
    buf: list[dict] = []
    buf_len = 0

    def flush() -> list[dict]:
        nonlocal buf, buf_len
        if buf_len >= min_chars:
            chunks.append({
                "text": "".join(s["_clean"] for s in buf),
                "start_ts": buf[0]["start"],
                "end_ts": buf[-1]["end"],
            })
        # 回退保留尾部句段作为下一块开头（重叠）
        tail, tlen = [], 0
        for s in reversed(buf):
            if tlen >= overlap:
                break
            tail.insert(0, s)
            tlen += len(s["_clean"])
        buf = tail
        buf_len = tlen
        return buf

    for seg in segments:
        seg = {**seg, "_clean": clean(seg["text"])}
        if not seg["_clean"]:
            continue
        buf.append(seg)
        buf_len += len(seg["_clean"])
        if buf_len >= max_chars:
            flush()
    # 收尾（避免重叠回退导致重复 flush，最后单独处理）
    if buf_len >= min_chars and (not chunks or buf[-1]["end"] != chunks[-1]["end_ts"]):
        chunks.append({
            "text": "".join(s["_clean"] for s in buf),
            "start_ts": buf[0]["start"],
            "end_ts": buf[-1]["end"],
        })
    return chunks


def main() -> None:
    cfg = load_config()
    tdir = resolve(cfg["paths"]["transcripts"])
    cdir = resolve(cfg["paths"]["chunks"])
    cdir.mkdir(parents=True, exist_ok=True)
    lexicon = load_lexicon(cfg)
    ch = cfg["chunking"]
    match_code = cfg["entity"]["match_stock_code"]

    files = sorted(tdir.glob("*.json"))
    if not files:
        raise SystemExit(f"{tdir} 下没有转写文件，请先跑 transcribe")

    total_chunks = 0
    for tf in tqdm(files, desc="分段"):
        doc = json.loads(tf.read_text(encoding="utf-8"))
        raw_chunks = chunk_segments(
            doc.get("segments", []), ch["max_chars"], ch["overlap_chars"], ch["min_chars"]
        )
        out = []
        for i, c in enumerate(raw_chunks):
            out.append({
                "chunk_id": f"{doc['video_id']}::{i}",
                "video_id": doc["video_id"],
                "title": doc["title"],
                "day_index": doc.get("day_index"),
                "start_ts": round(c["start_ts"], 1),
                "end_ts": round(c["end_ts"], 1),
                "text": c["text"],
                "stocks": extract_entities(c["text"], lexicon, match_code),
            })
        (cdir / f"{doc['video_id']}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total_chunks += len(out)

    print(f"完成 {len(files)} 个转写 → {total_chunks} 个 chunk，输出 → {cdir}")


if __name__ == "__main__":
    main()
