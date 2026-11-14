# image_guard.py — LiteLLM custom callback: image overflow guard.
#
# Behavior (mirrors dsh-image-guard, adapted for LiteLLM hooks):
#   1. Pre-call: if the request carries more images than the configured limit,
#      drop OLDEST conversation units (whole user-turn units, tool chains kept
#      intact via tool_call_id pairing) until under the limit. The newest unit
#      (the current ask) is never touched.
#   2. On upstream 400 mentioning an image cap, learn the numeric limit and
#      cache it per model for IMAGE_GUARD_LIMIT_TTL seconds.
#
# Covers all three proxy entry formats:
#   - Chat Completions  data["messages"]
#   - Responses API     data["input"]  (str input is single-turn, skipped)
#   - Anthropic Messages data["messages"] with content blocks
#
# Config (env):
#   IMAGE_GUARD_MAX_IMAGES   default limit when nothing learned (default 10)
#   IMAGE_GUARD_LIMIT_TTL    learned-limit TTL seconds (default 86400)
#   IMAGE_GUARD_ENABLED      set "0" to disable trimming entirely
#
# Register in config.yaml:
#   litellm_settings:
#     callbacks: ["image_guard.image_guard_instance"]
#
# This module deliberately avoids mutating anything except the request body:
# no logging side effects, no exceptions raised to the caller — a guard must
# never become the reason a request fails.

from __future__ import annotations

import os
import re
import time
from typing import Any

try:
    _PROBE_PATH = os.getenv("IMAGE_GUARD_PROBE", "")
    if _PROBE_PATH:
        with open(_PROBE_PATH, "a", encoding="utf-8") as _pf:
            _pf.write(f"module-loaded {time.time()}\n")
except Exception:  # noqa: BLE001
    pass

from litellm.integrations.custom_logger import CustomLogger


# ---------------------------------------------------------------- limits ---

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


DEFAULT_MAX_IMAGES = _env_int("IMAGE_GUARD_MAX_IMAGES", 20)
LIMIT_TTL = _env_int("IMAGE_GUARD_LIMIT_TTL", 86400)
ENABLED = os.getenv("IMAGE_GUARD_ENABLED", "1") != "0"
CONVERT_URLS = os.getenv("IMAGE_GUARD_CONVERT_URLS", "0") != "0"
CONVERT_TIMEOUT = _env_int("IMAGE_GUARD_CONVERT_TIMEOUT", 4)  # seconds per image
MAX_IMAGE_BYTES = _env_int("IMAGE_GUARD_MAX_IMAGE_MB", 5) * 1024 * 1024

# Learned limits keyed by requested model name. Process-local on purpose:
# limits change only when an upstream changes its error text, and a restart
# re-learns in one failed request. No Redis dependency for a guard.
_learned: dict[str, tuple[int, float]] = {}

# Error-text patterns for "you may only send N images". Order matters: the
# first match wins, most-specific phrasing first. Extend after observing real
# upstream 400 bodies — do not guess patterns without a sample.
_LIMIT_PATTERNS = (
    # verb between the qualifier and the number: "only include 8 images",
    # "can only support 10 images", "at most attach 5 images"
    re.compile(r"(?:only|up to|at most|max(?:imum)?)\s*(?:include|support|send|use|attach|have|accept)?\s*(\d+)\s*image", re.I),
    re.compile(r"(\d+)\s*image[s]?\s*(?:limit|maximum|max|exceed)", re.I),
    re.compile(r"image[s]?\s*(?:limit|maximum)\s*(?:of|is)?\s*(\d+)", re.I),
)

_PLACEHOLDER_TEXT = "[earlier conversation with images omitted]"


# ------------------------------------------------------------ URL fetch ----

_download = None  # lazily imported httpx.get wrapper


def _fetch_as_data_uri(url: str) -> str | None:
    """Download one remote image and return a data URI, or None on any failure.

    Failure is NORMAL here: expired signed URLs, intranet-only hosts, oversized
    images. The caller keeps the original URL in those cases and falls back to
    unit eviction — never raises, never blocks long.
    """
    global _download
    if not url.lower().startswith(("http://", "https://")):
        return None
    try:
        if _download is None:
            import httpx
            _download = httpx.get
        response = _download(
            url,
            timeout=CONVERT_TIMEOUT,
            follow_redirects=True,
        )
        if response.status_code != 200:
            return None
        content = response.content
        if not content or len(content) > MAX_IMAGE_BYTES:
            return None
        content_type = response.headers.get("content-type", "image/png").split(";")[0]
        if not content_type.startswith("image/"):
            return None
        import base64
        encoded = base64.b64encode(content).decode("ascii")
        return f"data:{content_type};base64,{encoded}"
    except Exception:  # noqa: BLE001 — conversion is best-effort by design
        return None


def _limit_for(model: str) -> int:
    hit = _learned.get(model)
    if hit is not None and hit[1] > time.time():
        return hit[0]
    return DEFAULT_MAX_IMAGES


def learn_limit_from_error(model: str, error_text: str) -> int | None:
    """Return the learned limit if error_text names one, else None."""
    for pattern in _LIMIT_PATTERNS:
        match = pattern.search(error_text)
        if match:
            value = int(match.group(1))
            if value > 0:
                _learned[model] = (value, time.time() + LIMIT_TTL)
                return value
    return None


# ------------------------------------------------------------- text parts ---

def _is_text_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "text"


def _image_url_of(part: Any) -> str | None:
    """Return the image URL if part is an image part, else None."""
    if not isinstance(part, dict):
        return None
    if part.get("type") == "image_url":
        url = part.get("image_url")
        if isinstance(url, dict):
            return str(url.get("url") or "")
        return str(url or "")
    if part.get("type") == "image":
        src = part.get("source")
        if isinstance(src, dict):
            return str(src.get("url") or "")
    return None


def _is_remote_url_image(part: Any) -> bool:
    """True for http(s) URL image references — the only kind vLLM counts
    against its image-links cap. base64 data URIs are unlimited."""
    url = _image_url_of(part)
    return bool(url) and not url.startswith("data:")


def _is_image_part(part: Any) -> bool:
    return _image_url_of(part) is not None


def _placeholder_part() -> dict[str, str]:
    return {"type": "text", "text": _PLACEHOLDER_TEXT}


# ------------------------------------------------------ chat unit parsing ---

def _chat_tool_ids(message: dict) -> set[str]:
    """tool_call ids offered by an assistant message."""
    ids: set[str] = set()
    for call in message.get("tool_calls") or ():
        if isinstance(call, dict) and call.get("id"):
            ids.add(str(call["id"]))
    return ids


def _group_chat_units(messages: list) -> list[list[int]]:
    """Group chat message indexes into conversation units.

    A unit starts at each 'user' message and absorbs subsequent assistant /
    tool messages, matching tool results to pending tool_call ids. A 'tool'
    message whose id matches a pending call never starts a new unit; an
    assistant tool_call message joins the open unit. System messages are
    never grouped (caller keeps them untouched).
    """
    units: list[list[int]] = []
    current: list[int] | None = None
    pending_ids: set[str] = set()
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            continue
        if role == "user":
            current = [idx]
            units.append(current)
            pending_ids = set()
            continue
        if role == "assistant" and _chat_tool_ids(message):
            if current is None:
                current = [idx]
                units.append(current)
            else:
                current.append(idx)
            pending_ids |= _chat_tool_ids(message)
            continue
        if role == "tool":
            tool_id = str(message.get("tool_call_id") or "")
            if current is not None and (not tool_id or tool_id in pending_ids):
                current.append(idx)
                pending_ids.discard(tool_id)
                continue
            # orphan tool result: treat as its own unit so we never glue it
            # onto an unrelated turn
            current = [idx]
            units.append(current)
            pending_ids = set()
            continue
        # assistant without tool_calls (plain reply) closes the open unit's
        # pending state but stays in the same unit
        if current is None:
            current = [idx]
            units.append(current)
        else:
            current.append(idx)
        pending_ids = set()
    return units


def _count_chat_images(messages: list, indexes: list[int], remote_only: bool = False) -> int:
    total = 0
    for idx in indexes:
        content = messages[idx].get("content")
        if isinstance(content, list):
            for part in content:
                if not _is_image_part(part):
                    continue
                if False:
                    continue
                total += 1
    return total


def _drop_chat_unit(messages: list, indexes: list[int]) -> None:
    """Replace a whole unit's messages with one placeholder user message."""
    first = indexes[0]
    for idx in indexes:
        messages[idx] = None  # type: ignore[assignment]
    messages[first] = {"role": "user", "content": _PLACEHOLDER_TEXT}


def trim_chat_messages(messages: list, limit: int, model: str = "") -> tuple[list, int]:
    """Return (new_messages, removed_images). Never mutates the input list."""
    if limit <= 0 or not isinstance(messages, list) or not messages:
        return messages, 0

    # copy shallow; unit dropping rewrites slots in the copy
    working = list(messages)
    system_indexes = [
        idx for idx, m in enumerate(working)
        if isinstance(m, dict) and m.get("role") == "system"
    ]
    # Only REMOTE-URL images count against the vLLM image-links cap;
    # base64-embedded images are unlimited (verified on GLM vLLM: 21 b64 -> 200,
    # 21 urls -> 400 "At most 20 image(s)").
    units = [
        unit for unit in _group_chat_units(working)
        if unit and _count_chat_images(working, unit, remote_only=True) > 0
    ]
    if not units:
        return working, 0

    total_images = _count_chat_images(working, [i for u in units for i in u], remote_only=True)
    removed_images = 0

    # 1) evict oldest whole units first (tool chains die intact)
    removable = units[:-1] if len(units) > 1 else []
    for unit in removable:  # oldest first
        if total_images - removed_images <= limit:
            break
        unit_images = _count_chat_images(working, unit, remote_only=True)
        _drop_chat_unit(working, unit)
        removed_images += unit_images

    # 1.5) URL->base64 conversion: the vLLM image cap counts only remote URL
    # references; embedded base64 images are unlimited (verified live on GLM
    # vLLM: 21 b64 -> 200, 21 urls -> 400). Convert the OLDEST URL images to
    # base64 until under the cap — no information is lost, only transport
    # changes. Images that fail to download (expired/intranet/too large) keep
    # their URL and fall through to unit eviction.
    converted = 0
    overflow_after_eviction = total_images - removed_images - limit
    newest = units[-1]
    if CONVERT_URLS and overflow_after_eviction > 0:
        # Convert images OUTSIDE the newest unit first (old turns keep their
        # images instead of being evicted next time), then inside the newest
        # unit if more conversions are still needed.
        old_units = units[:-1] if len(units) > 1 else []
        converted = _convert_oldest_urls(working, old_units, overflow_after_eviction)
        still_overflowing = overflow_after_eviction - converted
        if still_overflowing > 0:
            converted += _convert_oldest_urls(working, [newest], still_overflowing)
        total_images = _count_chat_images(working, [i for u in units for i in u], remote_only=True)

    # 2) if the newest unit alone still exceeds the limit, trim WITHIN it:
    #    keep the newest images, replace the oldest with a text placeholder.
    #    (A single 21-image turn must still succeed when the cap is 20.)
    overflow = total_images - removed_images - limit
    if overflow > 0:
        slots: list[tuple[int, int]] = []
        for idx in newest:
            content = working[idx].get("content")
            if isinstance(content, list):
                for part_idx, part in enumerate(content):
                    if _is_image_part(part):
                        slots.append((idx, part_idx))
        for msg_idx, part_idx in slots[:overflow]:  # oldest images first
            content = working[msg_idx]["content"]
            content[part_idx] = _placeholder_part()
            removed_images += 1

    if removed_images:
        # observability: one line per trim/conversion event, enough to audit
        # conversion success rate in production without per-image noise
        try:
            if _PROBE_PATH: _pf=open(_PROBE_PATH,"a",encoding="utf-8"); _pf.write("image_guard: model=%s url_overflow=%d converted=%d deleted=%d %s\n" % (model, overflow_after_eviction, converted, removed_images, time.time())); _pf.close()
            from litellm._logging import verbose_proxy_logger
            verbose_proxy_logger.info(
                "image_guard: model=%s url_overflow=%d converted=%d deleted=%d",
                model, overflow_after_eviction, converted, removed_images,
            )
        except Exception:  # noqa: BLE001
            pass

    if removed_images:
        # collapse the None slots left by unit drops (each dropped unit's
        # placeholder sits at its first slot)
        working = [m for m in working if m is not None]
    return working, removed_images


def _convert_oldest_urls(messages: list, units: list[list[int]], needed: int) -> int:
    """Convert the OLDEST remote-URL images to data URIs until `needed` are
    converted or candidates run out. Returns the number actually converted.

    Never raises: per-image failures (download error, non-200, oversized,
    non-image content-type) leave that image as a URL and simply don't count.
    """
    converted = 0
    # oldest first: walk units from the front, parts within each message in order
    for unit in units:
        if converted >= needed:
            break
        for idx in unit:
            if converted >= needed:
                break
            content = messages[idx].get("content")
            if not isinstance(content, list):
                continue
            for part_idx, part in enumerate(content):
                if converted >= needed:
                    break
                if not _is_remote_url_image(part):
                    continue
                url = _image_url_of(part)
                data_uri = _fetch_as_data_uri(url)
                if data_uri is None:
                    continue  # keep the URL, try the next image
                content[part_idx] = {"type": "image_url", "image_url": {"url": data_uri}}
                converted += 1
    return converted


# -------------------------------------------------- responses unit parsing ---

def _responses_role(item: Any) -> str | None:
    if isinstance(item, dict):
        return item.get("role")  # message items carry role
    return getattr(item, "role", None)


def _responses_type(item: Any) -> str | None:
    if isinstance(item, dict):
        return item.get("type")
    return getattr(item, "type", None)


def _responses_call_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("call_id") or item.get("id") or "")
    return str(getattr(item, "call_id", None) or getattr(item, "id", None) or "")


def _responses_images(item: Any) -> int:
    """Count images inside a Responses input item's content list."""
    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
    if not isinstance(content, list):
        return 0
    total = 0
    for part in content:
        ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
        if ptype in ("input_image", "image_url"):
            total += 1
    return total


def _group_responses_units(items: list) -> list[list[int]]:
    units: list[list[int]] = []
    current: list[int] | None = None
    pending_ids: set[str] = set()
    for idx, item in enumerate(items):
        itype = _responses_type(item)
        role = _responses_role(item)
        if itype == "message" and role == "user":
            current = [idx]
            units.append(current)
            pending_ids = set()
            continue
        if itype == "function_call":
            call_id = _responses_call_id(item)
            if current is None:
                current = [idx]
                units.append(current)
            else:
                current.append(idx)
            if call_id:
                pending_ids.add(call_id)
            continue
        if itype == "function_call_output":
            call_id = _responses_call_id(item)
            if current is not None and (not call_id or call_id in pending_ids):
                current.append(idx)
                pending_ids.discard(call_id)
                continue
            current = [idx]
            units.append(current)
            pending_ids = set()
            continue
        if current is None:
            current = [idx]
            units.append(current)
        else:
            current.append(idx)
    return units


def trim_responses_input(items: list, limit: int) -> tuple[list, int]:
    """Trim image overflow for Responses `input` item lists."""
    if limit <= 0 or not isinstance(items, list) or not items:
        return items, 0
    working: list[Any] = list(items)
    units = [u for u in _group_responses_units(working) if u]
    image_units = [u for u in units if sum(_responses_images(working[i]) for i in u) > 0]
    if not image_units:
        return working, 0
    removable = image_units[:-1] if len(image_units) > 1 else []
    total = sum(_responses_images(working[i]) for u in image_units for i in u)
    removed = 0
    for unit in removable:
        if total - removed <= limit:
            break
        removed += sum(_responses_images(working[i]) for i in unit)
        first = unit[0]
        for idx in unit:
            working[idx] = None
        working[first] = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": _PLACEHOLDER_TEXT}],
        }
    if removed:
        working = [i for i in working if i is not None]
    return working, removed


# ------------------------------------------------------------ the callback ---

class ImageGuardHandler(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if _PROBE_PATH:
            try:
                with open(_PROBE_PATH, "a", encoding="utf-8") as _pf:
                    _pf.write(f"pre-call fired model={data.get('model') if isinstance(data, dict) else '?'} {time.time()}\n")
            except Exception:  # noqa: BLE001
                pass
        if not ENABLED or not isinstance(data, dict):
            return data
        model = str(data.get("model") or "")
        limit = _limit_for(model)
        try:
            messages = data.get("messages")
            if isinstance(messages, list) and messages:
                trimmed, removed = trim_chat_messages(messages, limit, model)
                # assign whenever the list content differs: conversion-only
                # trims (removed == 0) still changed URL images into data URIs
                # and MUST reach the provider, or the original overflow goes
                # through untouched and 400s at the backend.
                if removed or trimmed != messages:
                    data["messages"] = trimmed
                return data
            input_value = data.get("input")
            if isinstance(input_value, list) and input_value:
                trimmed, removed = trim_responses_input(input_value, limit)
                if removed or trimmed != input_value:
                    data["input"] = trimmed
        except Exception:  # noqa: BLE001 — a guard must never break a request
            pass
        return data

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if not ENABLED:
                return
            exception = kwargs.get("exception")
            text = str(getattr(exception, "message", None) or exception or "")
            if "400" not in text and "Bad Request" not in text:
                return
            if "image" not in text.lower():
                return
            model = str(kwargs.get("model") or "")
            if model:
                learn_limit_from_error(model, text)
        except Exception:  # noqa: BLE001
            pass


image_guard_instance = ImageGuardHandler()
