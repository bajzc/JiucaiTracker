"""阿里云 OSS 上传 + 签名 URL，给 Paraformer 提供公网可访问的音频。

bucket 保持私有：上传后生成带签名的临时 GET URL 交给 DashScope，转写完可删。
"""
from __future__ import annotations

from pathlib import Path

from .common import require_env


def get_bucket(cfg: dict):
    import oss2

    o = cfg["asr"]["oss"]
    auth = oss2.Auth(
        require_env(o["access_key_id_env"]),
        require_env(o["access_key_secret_env"]),
    )
    return oss2.Bucket(auth, f"https://{o['endpoint']}", o["bucket"])


def object_key(cfg: dict, video_id: str) -> str:
    fmt = cfg.get("audio_format", "m4a")
    return f"{cfg['asr']['oss']['prefix']}{video_id}.{fmt}"


def upload(bucket, key: str, local_path: Path) -> None:
    bucket.put_object_from_file(key, str(local_path))


def signed_url(cfg: dict, bucket, key: str) -> str:
    expire = int(cfg["asr"]["oss"]["url_expire_seconds"])
    # slash_safe 保留路径分隔符；含中文的 key 会被正确编码
    return bucket.sign_url("GET", key, expire, slash_safe=True)


def delete(bucket, key: str) -> None:
    bucket.delete_object(key)
