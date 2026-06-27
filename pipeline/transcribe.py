"""阶段2：ASR 转写，输出带时间戳的句段 [{text, start, end}]（秒，可溯源）。

四个 provider（config.asr.provider）：
- mlx_whisper（默认，Apple Silicon GPU）：直接读 m4a，句级时间戳，可跑 large-v3 仍快。
- faster_whisper（纯 CPU 离线）：直接读 m4a，句级绝对时间戳，无需切片/OSS。
- dashscope（阿里云 Paraformer）：长音频原生，需先传 OSS。
- openai（whisper-1）：本地文件直传，但 >25MB 需 ffmpeg 切片+时间戳偏移（见下）。

热词：从 data/hotwords.txt（个股/术语）拼成 prompt/initial_prompt 提升专名识别。
幂等：已存在 data/transcripts/{video_id}.json 则跳过。

用法:
    python -m pipeline.transcribe              # 转写 manifest 中全部
    python -m pipeline.transcribe --overwrite  # 重新转写
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm

from .common import load_config, require_env, resolve


def load_hotwords(cfg: dict) -> str:
    hw_path = resolve(cfg["paths"]["hotwords"])
    if not hw_path.exists():
        return ""
    words = [w.strip() for w in hw_path.read_text(encoding="utf-8").splitlines() if w.strip()]
    # whisper prompt 大约 224 token 上限，取前面若干个即可
    return "、".join(words[:80])


def segment_audio(src: Path, out_dir: Path, seconds: int, fmt: str) -> list[Path]:
    """用 ffmpeg 把音频切成定长片段（16k 单声道），返回有序片段路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"seg_%04d.{fmt}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1",
         "-f", "segment", "-segment_time", str(seconds), "-loglevel", "error", pattern],
        check=True,
    )
    return sorted(out_dir.glob(f"seg_*.{fmt}"))


def transcribe_openai(client, model: str, audio: Path, prompt: str, language: str) -> list[dict]:
    """单个切片 → verbose_json，提取 segment 级时间戳。"""
    with open(audio, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
            language=language,
            prompt=prompt or None,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    segs = getattr(resp, "segments", None) or []
    return [{"text": s.text.strip(), "start": float(s.start), "end": float(s.end)} for s in segs]


def transcribe_file_openai(cfg: dict, audio: Path, prompt: str) -> list[dict]:
    from openai import OpenAI

    asr = cfg["asr"]
    client = OpenAI(
        base_url=asr.get("openai_base_url"),
        api_key=require_env(asr.get("openai_api_key_env", "OPENAI_API_KEY")),
    )
    seg_secs = int(asr["segment_seconds"])
    with tempfile.TemporaryDirectory() as td:
        pieces = segment_audio(audio, Path(td), seg_secs, asr["segment_format"])
        merged: list[dict] = []
        for idx, piece in enumerate(pieces):
            offset = idx * seg_secs
            for seg in transcribe_openai(client, asr["openai_model"], piece, prompt, asr["language"]):
                seg["start"] += offset
                seg["end"] += offset
                if seg["text"]:
                    merged.append(seg)
    return merged


def _write_transcript(out_dir: Path, rec: dict, segments: list[dict]) -> None:
    payload = {**rec, "segments": segments,
               "n_segments": len(segments),
               "duration": segments[-1]["end"] if segments else 0.0}
    (out_dir / f"{rec['video_id']}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pending(manifest: list[dict], out_dir: Path, overwrite: bool) -> list[dict]:
    if overwrite:
        return manifest
    return [r for r in manifest if not (out_dir / f"{r['video_id']}.json").exists()]


def run_mlx_whisper(cfg: dict, manifest: list[dict], out_dir: Path, overwrite: bool) -> None:
    """本地 mlx-whisper（Apple Silicon GPU）：直接读 m4a，逐句出绝对时间戳。"""
    import mlx_whisper

    mw = cfg["asr"]["mlx_whisper"]
    prompt = load_hotwords(cfg) or None
    todo = pending(manifest, out_dir, overwrite)
    done = failed = 0
    for rec in tqdm(todo, desc=f"ASR(mlx:{mw['model'].split('/')[-1]})"):
        try:
            res = mlx_whisper.transcribe(
                rec["audio_path"],
                path_or_hf_repo=mw["model"],
                language=cfg["asr"]["language"],
                initial_prompt=prompt,
            )
            segments = [{"text": s["text"].strip(),
                         "start": round(s["start"], 2), "end": round(s["end"], 2)}
                        for s in res["segments"] if s["text"].strip()]
        except Exception as e:  # 单个失败不中断整批
            failed += 1
            tqdm.write(f"[失败] {rec['video_id']}: {e}")
            continue
        _write_transcript(out_dir, rec, segments)
        done += 1
    print(f"转写完成 {done}，跳过 {len(manifest) - len(todo)}，失败 {failed}，输出 → {out_dir}")


def run_faster_whisper(cfg: dict, manifest: list[dict], out_dir: Path, overwrite: bool) -> None:
    """本地 faster-whisper：模型只加载一次，直接读 m4a，逐句出绝对时间戳。"""
    from faster_whisper import WhisperModel

    fw = cfg["asr"]["faster_whisper"]
    prompt = load_hotwords(cfg) or None
    model = WhisperModel(fw["model"], device=fw["device"], compute_type=fw["compute_type"])

    todo = pending(manifest, out_dir, overwrite)
    done = failed = 0
    for rec in tqdm(todo, desc=f"ASR(faster-whisper:{fw['model']})"):
        try:
            segs_gen, _info = model.transcribe(
                rec["audio_path"],
                language=cfg["asr"]["language"],
                initial_prompt=prompt,
                vad_filter=fw["vad_filter"],
                beam_size=fw["beam_size"],
            )
            segments = [{"text": s.text.strip(), "start": round(s.start, 2), "end": round(s.end, 2)}
                        for s in segs_gen if s.text.strip()]
        except Exception as e:  # 单个失败不中断整批
            failed += 1
            tqdm.write(f"[失败] {rec['video_id']}: {e}")
            continue
        _write_transcript(out_dir, rec, segments)
        done += 1
    print(f"转写完成 {done}，跳过 {len(manifest) - len(todo)}，失败 {failed}，输出 → {out_dir}")


def run_openai(cfg: dict, manifest: list[dict], out_dir: Path, overwrite: bool) -> None:
    prompt = load_hotwords(cfg)
    done = failed = 0
    todo = pending(manifest, out_dir, overwrite)
    for rec in tqdm(todo, desc="ASR(openai)"):
        try:
            segments = transcribe_file_openai(cfg, Path(rec["audio_path"]), prompt)
        except Exception as e:  # 单个失败不中断整批
            failed += 1
            tqdm.write(f"[失败] {rec['video_id']}: {e}")
            continue
        _write_transcript(out_dir, rec, segments)
        done += 1
    print(f"转写完成 {done}，跳过 {len(manifest) - len(todo)}，失败 {failed}，输出 → {out_dir}")


def run_dashscope(cfg: dict, manifest: list[dict], out_dir: Path, overwrite: bool) -> None:
    """上传 OSS → 批量 Paraformer → 写转写 → 删 OSS 对象。"""
    from . import oss
    from .paraformer import transcribe_batch

    todo = pending(manifest, out_dir, overwrite)
    bucket = oss.get_bucket(cfg)
    batch_size = int(cfg["asr"]["batch_size"])
    delete_after = cfg["asr"]["oss"]["delete_after"]
    done = failed = 0

    for i in tqdm(range(0, len(todo), batch_size), desc="ASR(paraformer)"):
        batch = todo[i : i + batch_size]
        items, keys = [], {}
        for rec in batch:  # 上传并签名
            key = oss.object_key(cfg, rec["video_id"])
            try:
                oss.upload(bucket, key, Path(rec["audio_path"]))
            except Exception as e:
                failed += 1
                tqdm.write(f"[上传失败] {rec['video_id']}: {e}")
                continue
            url = oss.signed_url(cfg, bucket, key)
            items.append((rec["video_id"], url))
            keys[rec["video_id"]] = key
        if not items:
            continue
        try:
            results = transcribe_batch(cfg, items)
        except Exception as e:
            failed += len(items)
            tqdm.write(f"[转写失败] batch {i}: {e}")
            results = {}
        rec_by_id = {r["video_id"]: r for r in batch}
        for vid, segs in results.items():
            if segs is None:
                failed += 1
                tqdm.write(f"[识别失败] {vid}")
                continue
            _write_transcript(out_dir, rec_by_id[vid], segs)
            done += 1
        if delete_after:
            for key in keys.values():
                try:
                    oss.delete(bucket, key)
                except Exception:
                    pass

    print(f"转写完成 {done}，跳过 {len(manifest) - len(todo)}，失败 {failed}，输出 → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    manifest = json.loads(resolve(cfg["paths"]["manifest"]).read_text(encoding="utf-8"))
    out_dir = resolve(cfg["paths"]["transcripts"])
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = cfg["asr"]["provider"]
    {
        "mlx_whisper": run_mlx_whisper,
        "faster_whisper": run_faster_whisper,
        "openai": run_openai,
        "dashscope": run_dashscope,
    }[provider](cfg, manifest, out_dir, args.overwrite)


if __name__ == "__main__":
    main()
