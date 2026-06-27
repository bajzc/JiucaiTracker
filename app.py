"""炒股视频 RAG 问答 —— CLI 入口。

    python app.py                      # 交互问答
    python app.py -q "白云山的买卖逻辑是什么？"   # 单次提问

流程：检索(dense+过滤+rerank) → 拼接带出处的上下文 → LLM 生成带引用答案。
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from pipeline.common import dbg, is_debug, load_config, require_env, set_debug
from rag.prompts import build_messages
from rag.query import retrieve, rewrite_query

console = Console()


def llm_answer(messages: list[dict], cfg: dict) -> str:
    from openai import OpenAI

    lc = cfg["llm"]
    if is_debug():
        dbg(f"LLM 请求 · {lc['model']}  (temperature={lc['temperature']})",
            "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages))
    client = OpenAI(base_url=lc["base_url"], api_key=require_env(lc["api_key_env"]))
    resp = client.chat.completions.create(
        model=lc["model"], messages=messages, temperature=lc["temperature"],
    )
    reply = resp.choices[0].message.content
    if is_debug():
        usage = getattr(resp, "usage", None)
        dbg("LLM 响应", reply + (f"\n\n[tokens] {usage}" if usage else ""))
    return reply


def answer(question: str, cfg: dict, history: list[dict] | None = None) -> str | None:
    retrieval_query = rewrite_query(question, history or [], cfg)
    contexts = retrieve(retrieval_query, cfg)
    if not contexts:
        console.print("[yellow]资料中未检索到相关内容。[/yellow]")
        return None
    reply = llm_answer(build_messages(question, contexts, history), cfg)
    console.print(Panel(Markdown(reply), title="回答", border_style="green"))
    console.print("[dim]检索到的出处：[/dim]")
    for i, c in enumerate(contexts, 1):
        console.print(f"  [cyan][{i}][/cyan] {c.cite()}  "
                      f"[dim]相关度={c.score:.3f} 个股={c.stocks}[/dim]")
    return reply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", help="单次提问；不给则进入交互模式")
    ap.add_argument("--debug", action="store_true", help="打印每次模型调用与 prompt")
    args = ap.parse_args()
    if args.debug:
        set_debug(True)
    cfg = load_config()

    if args.question:
        answer(args.question, cfg)
        return

    console.print("[bold]炒股视频问答[/bold]（输入问题，Ctrl-C / 空行退出）")
    history: list[dict] = []
    try:
        while True:
            q = console.input("\n[bold cyan]问> [/bold cyan]").strip()
            if not q:
                break
            reply = answer(q, cfg, history)
            if reply is not None:
                history.append({"role": "user", "content": q})
                history.append({"role": "assistant", "content": reply})
    except (KeyboardInterrupt, EOFError):
        console.print("\n再见。")


if __name__ == "__main__":
    main()
