"""可选：用 data/hotwords.txt 创建 Paraformer 热词表，打印 vocabulary_id。

把输出的 id 填进 config.yaml 的 asr.vocabulary_id，转写时即生效，
提升个股名/术语等专名识别准确率。

    python -m pipeline.make_vocabulary
"""
from __future__ import annotations

from .common import load_config, require_env, resolve
from .transcribe import load_hotwords  # 复用同一份热词来源（取前若干）


def main() -> None:
    import dashscope
    from dashscope.audio.asr import VocabularyService

    cfg = load_config()
    dashscope.api_key = require_env(cfg["asr"]["dashscope_api_key_env"])

    # 直接读热词文件成列表（load_hotwords 返回的是拼接串，这里重新读）
    hw_path = resolve(cfg["paths"]["hotwords"])
    words = [w.strip() for w in hw_path.read_text(encoding="utf-8").splitlines()
             if w.strip() and not w.startswith("#")]
    if not words:
        raise SystemExit(f"{hw_path} 里没有热词")

    vocabulary = [{"text": w, "weight": 4, "lang": "zh"} for w in words]
    service = VocabularyService()
    vocab_id = service.create_vocabulary(
        prefix="jiucai", target_model=cfg["asr"]["dashscope_model"], vocabulary=vocabulary)

    print(f"已创建热词表，共 {len(words)} 词。")
    print(f"vocabulary_id = {vocab_id}")
    print("→ 请把它填进 config.yaml 的 asr.vocabulary_id")


if __name__ == "__main__":
    main()
