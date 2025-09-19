from __future__ import annotations
import json
from typing import Any, AsyncIterable


def extract_content(resp: Any) -> str:
    if isinstance(resp, dict):
        choices = resp.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content", "") or ""
        return ""
    choices = getattr(resp, "choices", None)
    if choices:
        first = choices[0]
        msg = getattr(first, "message", None)
        if msg is not None:
            return getattr(msg, "content", "") or ""
    return ""


async def assemble_stream_text(chunks: AsyncIterable[Any]) -> str:
    out: list[str] = []
    async for c in chunks:
        if isinstance(c, dict):
            choices = c.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    out.append(text)
        else:
            choices = getattr(c, "choices", None)
            if choices:
                delta = getattr(choices[0], "delta", None)
                if delta is not None:
                    text = getattr(delta, "content", "") or ""
                    if text:
                        out.append(text)
    return "".join(out)


def augment_json_with_cost(obj_json: str, resp: Any) -> str:
    try:
        obj = json.loads(obj_json)
    except Exception:
        obj = {"raw": obj_json}
    meta = {}
    usage = getattr(resp, "usage", None)
    if isinstance(usage, dict):
        meta["usage"] = usage
    elif hasattr(usage, "model_dump"):
        meta["usage"] = usage.model_dump()
    hidden = getattr(resp, "_hidden_params", None)
    if isinstance(hidden, dict):
        meta.update(hidden)
    obj["metadata"] = meta
    return json.dumps(obj)
