"""阶段1：扫描 audios/ 下的音频，解析文件名元数据，输出 manifest.json。

已有 972 个 .m4a，无需抽取/转码。文件名即标题，内含“第N日/天”序号，
作为按时间排序的代理键。

用法:
    python -m pipeline.scan_audio            # 全部
    python -m pipeline.scan_audio --limit 10 # 试点：只取前 10 个
"""
from __future__ import annotations

import argparse
import json

from .common import load_config, parse_metadata, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只扫描前 N 个（试点用）")
    args = ap.parse_args()

    cfg = load_config()
    audio_dir = resolve(cfg["paths"]["audio"])
    fmt = cfg.get("audio_format", "m4a")

    files = sorted(audio_dir.glob(f"*.{fmt}"))
    if not files:
        raise SystemExit(f"在 {audio_dir} 下没找到 *.{fmt} 文件")

    records = [parse_metadata(p) for p in files]
    # 按交易日序号降序：最近的在前（无序号的排最后）。
    # 这样 --limit N 取的是“最近 N 天”，更适合验证当前风格的内容。
    records.sort(key=lambda r: (r["day_index"] is not None, r["day_index"] or 0), reverse=True)
    if args.limit:
        records = records[: args.limit]

    manifest_path = resolve(cfg["paths"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_day = sum(1 for r in records if r["day_index"] is not None)
    print(f"扫描到 {len(files)} 个音频，写入 {len(records)} 条到 {manifest_path}")
    print(f"其中可解析“第N日/天”序号: {n_day}/{len(records)}")
    print("示例:")
    for r in records[:5]:
        print(f"  day={r['day_index']!s:>5}  {r['title'][:50]}")


if __name__ == "__main__":
    main()
