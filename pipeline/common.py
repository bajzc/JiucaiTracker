"""共享工具：配置加载、路径、API 客户端、文件名元数据解析、debug 日志。"""
from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# --- debug 开关：打印每次模型调用与 prompt（环境变量 JIUCAI_DEBUG 或 app.py --debug）---
_DEBUG = os.environ.get("JIUCAI_DEBUG", "").lower() not in ("", "0", "false", "no")


def set_debug(on: bool) -> None:
    global _DEBUG
    _DEBUG = on


def is_debug() -> bool:
    return _DEBUG


def dbg(title: str, body: str = "") -> None:
    """打到 stderr，避免和 stdout 上的正式回答混在一起。"""
    if not _DEBUG:
        return
    bar = "─" * 10
    print(f"\n{bar} [DEBUG] {title} {bar}", file=sys.stderr, flush=True)
    if body:
        print(body, file=sys.stderr, flush=True)


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel: str) -> Path:
    """把 config 里的相对路径解析为绝对路径。"""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"缺少环境变量 {name}。请复制 .env.example 为 .env 并填入真实 key。"
        )
    return val


# --- 文件名元数据解析 ---
# 标题形如：
#   第1044日投资记录：市场继续反弹，平仓白云山，增加安井H的仓位。
#   大学生第179日实盘炒股，头部效应明显，赛道走弱
#   【大学生10万炒股】第二十一天实盘
# 优先匹配“第N日/天”，再兜底无“第”的日记式命名（598日 / 564.实盘 / day.567）
_DAY_PATTERNS = [
    re.compile(r"第\s*(\d+)\s*[日天]"),
    re.compile(r"(?:^|[】：:\.\s])(\d{2,4})\s*日[实投]"),  # 598日实盘 / 1204日投资
    re.compile(r"(?i)day[\.\s]*(\d{2,4})"),               # day.567
    re.compile(r"(?:^|[】\s])(\d{2,4})[\.．]\s*实盘"),      # 564.实盘
    re.compile(r"(\d{2,4})\s*日\s*实"),                     # 大学生206日实盘
    re.compile(r"第\s*(\d{2,4})"),                          # 第352实盘 / 第711&712日（取首个）
]
# 中文数字日（如“第二十一天”）的兜底映射
_CN_NUM = {c: i for i, c in enumerate("零一二三四五六七八九")}


def _cn_to_int(s: str) -> int | None:
    if not s:
        return None
    if s.isdigit():
        return int(s)
    total, section, num = 0, 0, 0
    units = {"十": 10, "百": 100, "千": 1000}
    for ch in s:
        if ch in _CN_NUM:
            num = _CN_NUM[ch]
        elif ch in units:
            section += (num or 1) * units[ch]
            num = 0
        else:
            return None
    return total + section + num


def parse_metadata(audio_path: Path) -> dict[str, Any]:
    """从音频文件名提取标题与“第N日/天”序号（用作时间序代理）。"""
    title = audio_path.stem
    day_index = None
    for pat in _DAY_PATTERNS:
        m = pat.search(title)
        if m:
            day_index = int(m.group(1))
            break
    if day_index is None:
        m = re.search(r"第([零一二三四五六七八九十百千]+)[日天]", title)
        if m:
            day_index = _cn_to_int(m.group(1))
    return {
        "video_id": audio_path.stem,  # 文件名即唯一 id
        "title": title,
        "day_index": day_index,
        "audio_path": str(audio_path),
    }
