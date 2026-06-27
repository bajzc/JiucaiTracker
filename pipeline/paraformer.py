"""阿里云 Paraformer-v2 录音文件识别：批量提交 OSS URL，取回句级时间戳。

Paraformer 原生支持长音频（无需切片），返回的 sentences 自带绝对时间戳，
比 whisper 路线少了切片+偏移这一步。
"""
from __future__ import annotations

from http import HTTPStatus
from urllib.parse import urlparse

import requests

from .common import require_env


def _path(url: str) -> str:
    return urlparse(url).path


def parse_result(data: dict) -> list[dict]:
    """transcription_url 指向的 JSON → [{text, start, end}]（秒）。"""
    segs: list[dict] = []
    for tr in data.get("transcripts", []):
        for s in tr.get("sentences", []):
            text = (s.get("text") or "").strip()
            if text:
                segs.append({
                    "text": text,
                    "start": s["begin_time"] / 1000.0,
                    "end": s["end_time"] / 1000.0,
                })
    segs.sort(key=lambda x: x["start"])
    return segs


def transcribe_batch(cfg: dict, items: list[tuple[str, str]]) -> dict[str, list[dict] | None]:
    """items: [(video_id, signed_url)]；返回 {video_id: segments 或 None(失败)}。"""
    import dashscope
    from dashscope.audio.asr import Transcription

    asr = cfg["asr"]
    dashscope.api_key = require_env(asr["dashscope_api_key_env"])
    urls = [u for _, u in items]
    by_path = {_path(u): vid for vid, u in items}

    kwargs = dict(
        model=asr["dashscope_model"],
        file_urls=urls,
        language_hints=[asr["language"]],
    )
    if asr.get("vocabulary_id"):
        kwargs["vocabulary_id"] = asr["vocabulary_id"]

    task = Transcription.async_call(**kwargs)
    resp = Transcription.wait(task=task.output.task_id)
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"Paraformer 任务失败: {resp.output}")

    out: dict[str, list[dict] | None] = {}
    for r in resp.output["results"]:
        # 按 URL 路径匹配回 video_id（签名 URL 带 query，故只比 path）
        vid = by_path.get(_path(r["file_url"]))
        if vid is None:
            continue
        if r.get("subtask_status") != "SUCCEEDED":
            out[vid] = None
            continue
        data = requests.get(r["transcription_url"], timeout=120).json()
        out[vid] = parse_result(data)
    return out
