# test_image_guard.py — deterministic unit tests for image_guard trimming.
# Run: python test_image_guard.py   (exit 0 = all pass)
#
# Intent (WHY these tests): a guard that deletes the wrong messages breaks
# agent conversations silently — worse than the 400 it prevents. Each test
# pins one contract we rely on in production:
#   1. newest image-bearing unit survives (current ask untouched)
#   2. tool chains die whole (no dangling tool_call / tool result)
#   3. system messages never touched
#   4. oldest-first eviction until under limit
#   5. malformed input passes through unchanged (guard never raises)

from __future__ import annotations

import sys

# Minimal litellm shim so the deterministic tests run without the package.
import types as _types

_litellm = _types.ModuleType("litellm")
_integrations = _types.ModuleType("litellm.integrations")
_custom_logger = _types.ModuleType("litellm.integrations.custom_logger")

class CustomLogger:
    pass

_custom_logger.CustomLogger = CustomLogger
_litellm.integrations = _integrations
_integrations.custom_logger = _custom_logger
import sys as _sys
_sys.modules["litellm"] = _litellm
_sys.modules["litellm.integrations"] = _integrations
_sys.modules["litellm.integrations.custom_logger"] = _custom_logger

from image_guard import (  # noqa: E402
    _group_chat_units,
    learn_limit_from_error,
    trim_chat_messages,
    trim_responses_input,
)


def img(idx: int, url: bool = True) -> dict:
    if url:
        return {"type": "image_url", "image_url": {"url": f"https://x/img{idx}.png"}}
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,AAAA{idx}"}}


def user_turn(text: str, images: int, start_idx: int = 0) -> dict:
    parts: list = [{"type": "text", "text": text}]
    for i in range(images):
        parts.append(img(start_idx + i))
    return {"role": "user", "content": parts}


def chat_chain(turn_text: str, images: int, call_id: str, start_idx: int = 0) -> list:
    """user(+images) -> assistant(tool_call) -> tool(result) -> assistant(reply)"""
    return [
        user_turn(turn_text, images, start_idx),
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": call_id, "type": "function",
                         "function": {"name": "run", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": "result"},
        {"role": "assistant", "content": f"done for {call_id}"},
    ]


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def count_images(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            total += sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
    return total


# ---- unit grouping ----
def test_grouping() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        *chat_chain("look", 1, "c1", 0),
        *chat_chain("again", 1, "c2", 1),
    ]
    units = _group_chat_units(messages)
    # system excluded; 2 image-bearing units; sizes 4 and 4
    sizes = [len(u) for u in units]
    check("grouping: tool chain stays in one unit", sizes == [4, 4], f"sizes={sizes}")


# ---- newest unit protection ----
def test_newest_protected() -> None:
    messages = [
        *chat_chain("old turn", 3, "c1", 0),
        user_turn("current ask", 2, 3),
    ]
    trimmed, removed = trim_chat_messages(messages, limit=2)
    check("newest unit images kept", count_images(trimmed) == 2, f"count={count_images(trimmed)}")
    check("removed count reported", removed == 3, f"removed={removed}")
    check("current ask text survives", any(
        isinstance(m.get("content"), list) and any(
            isinstance(p, dict) and p.get("type") == "text" and p.get("text") == "current ask"
            for p in m["content"]
        ) for m in trimmed
    ))


# ---- tool chain integrity ----
def test_tool_chain_whole() -> None:
    messages = [
        *chat_chain("old", 2, "c1", 0),
        user_turn("new", 1, 2),
    ]
    trimmed, _ = trim_chat_messages(messages, limit=1)
    # the old chain is fully replaced by a placeholder — no dangling call ids
    call_ids = [m.get("tool_calls", [{}])[0].get("id") for m in trimmed if m.get("tool_calls")]
    tool_refs = [m.get("tool_call_id") for m in trimmed if m.get("role") == "tool"]
    check("no dangling tool_call", all(cid in tool_refs for cid in call_ids) or not call_ids,
          f"calls={call_ids} tools={tool_refs}")
    check("placeholder present", any(m.get("content") == "[earlier conversation with images omitted]"
                                     for m in trimmed))


# ---- system untouched ----
def test_system_untouched() -> None:
    messages = [
        {"role": "system", "content": "you are helpful"},
        *chat_chain("old", 3, "c1", 0),
        user_turn("new", 1, 3),
    ]
    trimmed, _ = trim_chat_messages(messages, limit=1)
    check("system preserved verbatim", trimmed[0] == {"role": "system", "content": "you are helpful"})


# ---- oldest-first eviction, multiple units ----
def test_oldest_first() -> None:
    messages = [
        *chat_chain("t1", 2, "c1", 0),
        *chat_chain("t2", 2, "c2", 2),
        user_turn("t3", 2, 4),
    ]
    trimmed, removed = trim_chat_messages(messages, limit=4)
    # must evict exactly one full old unit (2 images) to reach 4
    check("evicts oldest only", removed == 2, f"removed={removed}")
    check("t2 survives", any(isinstance(m.get("content"), list) and any(
        isinstance(p, dict) and p.get("type") == "text" and p.get("text") == "t2"
        for p in m["content"]) for m in trimmed))


# ---- single-turn overflow: newest unit trimmed internally ----
def test_single_turn_overflow() -> None:
    # 21 images in ONE user turn, cap 20 -> 1 oldest image replaced, 20 kept
    parts = [{"type": "text", "text": "solo"}]
    for i in range(21):
        parts.append(img(i))
    messages = [{"role": "user", "content": parts}]
    trimmed, removed = trim_chat_messages(messages, limit=20)
    kept = count_images(trimmed)
    check("single-turn: evicts exactly 1", removed == 1, f"removed={removed}")
    check("single-turn: keeps newest 20", kept == 20, f"kept={kept}")
    check("single-turn: placeholder in place of oldest",
          any(isinstance(p, dict) and p.get("type") == "text"
              and p.get("text") == "[earlier conversation with images omitted]"
              for p in trimmed[0]["content"]))
    # newest image (index 20) must survive
    check("single-turn: newest image survives",
          any(isinstance(p, dict) and p.get("type") == "image_url"
              and "img20" in p.get("image_url", {}).get("url", "")
              for p in trimmed[0]["content"]))


# ---- base64 images ARE counted (verified live 2026-11-14) ----
def test_base64_counted() -> None:
    # GLM vLLM cap counts TOTAL images including data URIs: 21 DISTINCT b64
    # images return 400 "At most 20 image(s)". The guard must count them too.
    parts = [{"type": "text", "text": "solo"}]
    for i in range(21):
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,UNIQUE{i}"}})
    messages = [{"role": "user", "content": parts}]
    trimmed, removed = trim_chat_messages(messages, limit=20)
    total_imgs = sum(
        1 for p in trimmed[0]["content"]
        if isinstance(p, dict) and p.get("type") == "image_url"
    )
    check("b64 overflow: trimmed to 20 total images", total_imgs == 20, f"total={total_imgs}")
    # mixed 15 url + 10 b64 = 25 total -> trim 5 oldest images
    parts2 = [{"type": "text", "text": "mix"}]
    parts2 += [img(i) for i in range(15)]
    parts2 += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,BBBB{i}"}} for i in range(10)]
    messages2 = [{"role": "user", "content": parts2}]
    trimmed2, removed2 = trim_chat_messages(messages2, limit=20)
    total2 = sum(
        1 for p in trimmed2[0]["content"]
        if isinstance(p, dict) and p.get("type") == "image_url"
    )
    check("mixed url+b64: total trimmed to 20", total2 == 20, f"total={total2}")

# ---- conversion tier disabled by default ----
def test_conversion_disabled() -> None:
    """vLLM cap counts total images (b64 included), so conversion cannot
    bypass the cap. Guard must trim by placeholder regardless."""
    parts = [{"type": "text", "text": "t"}]
    for i in range(25):
        parts.append(img(i))
    messages = [{"role": "user", "content": parts}]
    trimmed, removed = trim_chat_messages(messages, limit=20)
    urls_left = sum(
        1 for p in trimmed[0]["content"]
        if isinstance(p, dict) and p.get("type") == "image_url"
        and not p["image_url"]["url"].startswith("data:")
    )
    check("no-conversion: 20 urls remain", urls_left == 20, f"urls={urls_left}")
    check("no-conversion: 5 images removed", removed == 5, f"removed={removed}")

# ---- under limit: no change ----
def test_noop_under_limit() -> None:
    messages = [*chat_chain("a", 1, "c1", 0), user_turn("b", 1, 1)]
    trimmed, removed = trim_chat_messages(messages, limit=5)
    check("under limit: identity", removed == 0 and trimmed == list(messages))


# ---- malformed input: never raises ----
def test_malformed() -> None:
    cases = [
        None,
        [],
        ["not-a-dict", 42, None],
        [{"role": "user"}, {"role": "assistant", "content": None}],
        [{"role": "user", "content": {"weird": "shape"}}],
    ]
    for i, case in enumerate(cases):
        try:
            trimmed, _ = trim_chat_messages(case if case is not None else [], limit=2)
            check(f"malformed case {i} survives", True)
        except Exception as exc:  # noqa: BLE001
            check(f"malformed case {i} survives", False, str(exc))


# ---- responses input ----
def test_responses() -> None:
    items = [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "old"}, {"type": "input_image", "image_url": "u1"}]},
        {"type": "function_call", "call_id": "fc1", "name": "run", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "fc1", "output": "ok"},
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "new"}, {"type": "input_image", "image_url": "u2"}]},
    ]
    trimmed, removed = trim_responses_input(items, limit=1)
    check("responses: oldest unit dropped", removed == 1, f"removed={removed}")
    check("responses: newest kept", any(
        isinstance(it, dict) and it.get("type") == "message" and isinstance(it.get("content"), list)
        and any(p.get("type") == "input_image" for p in it["content"])
        for it in trimmed
    ))
    # no dangling function_call without its output
    calls = [it.get("call_id") for it in trimmed if isinstance(it, dict) and it.get("type") == "function_call"]
    outs = [it.get("call_id") for it in trimmed if isinstance(it, dict) and it.get("type") == "function_call_output"]
    check("responses: call/output paired", all(c in outs for c in calls), f"calls={calls} outs={outs}")


# ---- limit learning ----
def test_learn() -> None:
    got = learn_limit_from_error("glm-chat",
        'litellm.BadRequestError: 400 - {"error": {"message": "You can only include 8 images."}}')
    check("learn: extracts 8", got == 8, f"got={got}")
    got_v = learn_limit_from_error("glm-chat",
        "BadRequestError: This model can only support 10 images per request.")
    check("learn: support phrasing", got_v == 10, f"got={got_v}")
    got2 = learn_limit_from_error("glm-chat", "some unrelated error")
    check("learn: unrelated error ignored", got2 is None)


# ---- hook assigns converted-only results (regression for U2c 400) ----
def test_hook_assigns_conversion_only() -> None:
    """removed==0 but URLs converted -> hook MUST swap data['messages'],
    else the provider sees the original overflow and 400s (seen live)."""
    import asyncio
    import image_guard as ig

    def fake_get(url, **kwargs):
        class R:
            status_code = 200
            content = b"PNG" + url[-10:].encode()
            headers = {"content-type": "image/png"}
        return R()

    orig_download = ig._download
    orig_learned = dict(ig._learned)
    ig._download = fake_get
    ig._learned.clear()  # production had learned limit=20; do the same
    ig._learned["glm-chat"] = (20, ig.time.time() + 3600)
    try:
        parts = [{"type": "text", "text": "describe"}]
        for i in range(21):
            parts.append({"type": "image_url", "image_url": {"url": f"https://x?seed={i}"}})
        data = {"model": "glm-chat", "messages": [{"role": "user", "content": parts}], "max_tokens": 30}
        handler = ig.ImageGuardHandler()
        out = asyncio.run(handler.async_pre_call_hook(None, None, data, "completion"))
        urls = sum(
            1 for p in out["messages"][0]["content"]
            if isinstance(p, dict) and p.get("type") == "image_url"
            and not p["image_url"]["url"].startswith("data:")
        )
        check("hook: conversion-only result is assigned", urls == 20, f"urls={urls}")
    finally:
        ig._download = orig_download
        ig._learned.clear()
        ig._learned.update(orig_learned)


if __name__ == "__main__":
    test_grouping()
    test_newest_protected()
    test_tool_chain_whole()
    test_system_untouched()
    test_oldest_first()
    test_single_turn_overflow()
    test_base64_counted()
    # URL conversion tier removed: vLLM cap counts total images (b64 included).

    test_conversion_disabled()
    test_noop_under_limit()
    test_malformed()
    test_responses()
    test_learn()
    test_hook_assigns_conversion_only()
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
        sys.exit(1)
    print("RESULT: ALL TESTS PASSED")
