from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.exceptions import BadRequestError, Timeout
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.common_utils.callback_utils import (
    get_metadata_variable_name_from_kwargs,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

log = logging.getLogger("straiker.guardrail")

DEFAULT_API_BASE = "https://api.prod.straiker.ai"
DEFAULT_MAX_PAYLOAD_BYTES = 524288
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


class DetectResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    score: float = Field(ge=0.0, le=1.0)
    turn_id: str | None = Field(default=None, alias="turnId")
    verdict: bool | None = None
    explanation: str | None = None
    debug: dict | None = None
    custom: dict | None = None


DetectResponse.model_rebuild()


def _extract_text_content(content: Any) -> str:
    texts: list[str] = []
    pending = [content]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            if value:
                texts.append(value)
            continue
        if isinstance(value, list):
            pending.extend(reversed(value))
            continue
        if not isinstance(value, dict):
            continue
        block_type = value.get("type")
        if block_type in ("text", "input_text", "output_text"):
            text = value.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
        elif block_type == "tool_result":
            pending.append(value.get("content"))
    return "\n".join(texts)


def _last_user_prompt(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return _extract_text_content(m.get("content"))
    return ""


def _has_scannable_input(messages: list[dict]) -> bool:
    return bool(_last_user_prompt(messages)) or any(
        message.get("role") == "tool" and bool(_extract_text_content(message.get("content")))
        for message in messages
        if isinstance(message, dict)
    )


def _get_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _transform_tool_calls(tcs: Any) -> list[dict] | None:
    if not isinstance(tcs, list):
        return None

    def transform(tc: Any) -> dict[str, Any]:
        fn = _get_field(tc, "function") or _get_field(tc, "func")
        name = _get_field(fn, "name") if fn is not None else _get_field(tc, "name")
        args = _get_field(fn, "arguments") if fn is not None else _get_field(tc, "input")
        if isinstance(args, str):
            try:
                tool_input = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                tool_input = {"_raw": args}
        else:
            tool_input = args
        return {"id": _get_field(tc, "id"), "name": name, "input": tool_input}

    return [transform(tc) for tc in tcs if isinstance(tc, dict) or hasattr(tc, "function")]


def _build_agentic_messages(req_messages: list[dict], app_response: str | None) -> list[dict]:
    out = []
    for m in req_messages or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        content_blocks = content if isinstance(content, list) else []
        tool_results = [
            block for block in content_blocks if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        native_tool_calls = [
            block for block in content_blocks if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        text = _extract_text_content(
            [
                block
                for block in content_blocks
                if not isinstance(block, dict) or block.get("type") not in ("tool_result", "tool_use")
            ]
            if content_blocks
            else content
        )
        existing_tool_calls = m.get("tool_calls")
        tool_calls = [
            *(existing_tool_calls if isinstance(existing_tool_calls, list) else []),
            *native_tool_calls,
        ]
        entry: dict[str, Any] = {"role": m.get("role")}
        if text:
            entry["content"] = text
        if tool_calls:
            entry["tool_calls"] = _transform_tool_calls(tool_calls)
        if m.get("tool_call_id"):
            entry["tool_call_id"] = m["tool_call_id"]
        if m.get("name"):
            entry["tool_name"] = m["name"]
        elif m.get("tool_name"):
            entry["tool_name"] = m["tool_name"]
        if not tool_results or len(entry) > 1:
            out.append(entry)
        out.extend(
            {
                "role": "tool",
                "content": _extract_text_content(tool_result.get("content")),
                "tool_call_id": tool_result.get("tool_use_id"),
            }
            for tool_result in tool_results
        )
    if app_response and app_response != "N/A":
        out.append({"role": "assistant", "content": app_response})
    return out


def _has_meaningful_tool_calls(tool_calls: Any) -> bool:
    if not isinstance(tool_calls, list) or not tool_calls:
        return False
    return any(bool(_get_field(_get_field(tc, "function"), "name") or _get_field(tc, "name")) for tc in tool_calls)


def _responses_output(response: Any) -> tuple[list[str], list[dict[str, Any]]]:
    output = _get_field(response, "output")
    if not isinstance(output, list):
        return [], []
    texts = [
        text
        for item in output
        if _get_field(item, "type") == "message"
        for block in (_get_field(item, "content") or [])
        if _get_field(block, "type") == "output_text"
        for text in [_get_field(block, "text")]
        if isinstance(text, str) and text
    ]
    tool_calls = [
        {
            "id": _get_field(item, "call_id") or _get_field(item, "id"),
            "function": {
                "name": _get_field(item, "name"),
                "arguments": _get_field(item, "arguments"),
            },
        }
        for item in output
        if _get_field(item, "type") == "function_call"
    ]
    return texts, tool_calls


def _structured_log(level: int, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    log.log(level, json.dumps(payload, default=str))


def _resolve_session_id(data: dict, meta: dict) -> str:
    requester_metadata = _as_dict(meta.get("requester_metadata"))
    return (
        meta.get("session_id")
        or requester_metadata.get("session_id")
        or data.get("litellm_call_id")
        or "litellm-session"
    )


def _resolve_user_name(data: dict, meta: dict) -> str:
    return meta.get("user_name") or data.get("user") or "litellm"


def _resolve_provider(data: dict, model: str | None) -> str | None:
    litellm_params = _as_dict(data.get("litellm_params"))
    custom_llm_provider = data.get("custom_llm_provider") or litellm_params.get("custom_llm_provider")
    if custom_llm_provider:
        return custom_llm_provider
    if not model:
        return None
    try:
        _, provider, _, _ = get_llm_provider(
            model=model,
            api_base=data.get("api_base") or litellm_params.get("api_base"),
            api_key=data.get("api_key") or litellm_params.get("api_key"),
        )
    except BadRequestError:
        return None
    return provider or None


def _redact_endpoint(endpoint: str) -> str:
    try:
        parsed = urlsplit(endpoint)
    except (UnicodeError, ValueError):
        return endpoint.split("#", 1)[0].split("?", 1)[0]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _resolve_destination(data: dict) -> str:
    litellm_params = _as_dict(data.get("litellm_params"))
    api_base = data.get("api_base") or litellm_params.get("api_base")
    if isinstance(api_base, str):
        try:
            hostname = urlsplit(api_base).hostname
        except ValueError:
            hostname = None
        if hostname:
            return hostname
    return "unknown"


def _extract_litellm_context(data: dict, meta: dict) -> dict:
    proxy_req = _as_dict(data.get("proxy_server_request"))
    identity_fields = {
        "key_alias": meta.get("user_api_key_alias"),
        "key_team": meta.get("user_api_key_team_alias") or meta.get("user_api_key_team_id"),
        "end_user": meta.get("user_api_key_end_user_id") or data.get("user"),
        "request_id": data.get("litellm_call_id") or meta.get("litellm_call_id"),
        "endpoint": proxy_req.get("url") or meta.get("endpoint"),
    }
    normalized_identity_fields = {
        key: _redact_endpoint(value) if key == "endpoint" and isinstance(value, str) else value
        for key, value in identity_fields.items()
    }
    context = {
        "temperature": data.get("temperature"),
        "max_tokens": data.get("max_tokens"),
        "stream": data.get("stream"),
        "call_type": data.get("call_type") or data.get("litellm_call_type"),
    }
    return {
        **{key: value for key, value in context.items() if value is not None},
        **normalized_identity_fields,
    }


class StraikerGuardrail(CustomGuardrail):
    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel]:
        from litellm.types.proxy.guardrails.guardrail_hooks.straiker import (
            StraikerGuardrailConfigModel,
        )

        return StraikerGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_key: str,
        api_base: str = DEFAULT_API_BASE,
        agentic: bool = False,
        source: str = "litellm",
        threshold: float = 0.5,
        timeout: float = 5.0,
        unreachable_fallback: str = "fail_closed",
        max_retries: int = 2,
        initial_backoff: float = 0.1,
        max_backoff: float = 2.0,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        custom_headers: dict[str, str] | None = None,
        verbose: bool = False,
        dedup_iterations: bool = True,
        async_handler: httpx.AsyncClient | None = None,
        **kwargs,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if unreachable_fallback not in ("fail_open", "fail_closed"):
            raise ValueError(f"unreachable_fallback must be 'fail_open' or 'fail_closed'; got {unreachable_fallback!r}")

        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.agentic = bool(agentic)
        self.source = source
        self.threshold = float(threshold)
        self.timeout = float(timeout)
        self.unreachable_fallback = unreachable_fallback
        self.max_retries = max(0, int(max_retries))
        self.initial_backoff = max(0.0, float(initial_backoff))
        self.max_backoff = max(self.initial_backoff, float(max_backoff))
        self.max_payload_bytes = int(max_payload_bytes)
        self.custom_headers = dict(custom_headers) if custom_headers else {}
        self.verbose = bool(verbose)
        self.dedup_iterations = bool(dedup_iterations)

        self.async_handler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(**kwargs)

    def _detect_url(self) -> str:
        suffix = "?agentic" if self.agentic else ""
        return f"{self.api_base}/api/v1/detect{suffix}"

    def _agent_source(self, agent_id: str | None) -> str:
        if agent_id:
            return agent_id
        return self.source

    def _build_payload(
        self,
        *,
        messages: list[dict],
        app_response: str,
        data: dict,
        hook: str,
        finish_reason: str | None = None,
    ) -> dict:
        metadata_field = get_metadata_variable_name_from_kwargs(data)
        metadata_value = data.get(metadata_field)
        meta = _as_dict(metadata_value)
        net = _as_dict(meta.get("network"))
        model = data.get("model", "unknown") or "unknown"
        litellm_params = _as_dict(data.get("litellm_params"))
        model_id = litellm_params.get("model") or model
        provider = _resolve_provider(data, model)
        destination = _resolve_destination(data)
        ctx = _extract_litellm_context(data, meta)
        agent_id = meta.get("agent_id")
        agent_source = self._agent_source(agent_id)
        session_id = _resolve_session_id(data, meta)
        user_name = _resolve_user_name(data, meta)
        user_role = meta.get("user_role") or "public"
        trace_id = meta.get("trace_id") or meta.get("litellm_trace_id") or data.get("litellm_trace_id")
        agent_role = meta.get("agent_role")
        request_id = data.get("litellm_call_id") or meta.get("litellm_call_id")
        ip = meta.get("requester_ip_address") or net.get("IP") or "127.0.0.1"
        ua = meta.get("user_agent") or net.get("User-Agent") or "litellm-proxy"
        event_type = "during_call" if hook == "moderation" else hook

        network = {"IP": ip, "User-Agent": ua, "Content-Type": "application/json"}
        hook_tag = f"litellm/{hook}"
        metadata = {
            "session_id": session_id,
            "user_name": user_name,
            "user_role": user_role,
            "client_ip": ip,
            "user_agent": ua,
            "app_name": agent_source,
            "gateway": "litellm",
            "integration": "litellm-straiker",
            "agent_id": agent_id,
            "agent_role": agent_role,
            "model_id": model_id,
            "model_provider": provider,
            "event_type": event_type,
            "litellm_hook": hook,
            "hook_tag": hook_tag,
            "trace_id": trace_id,
            "request_id": request_id,
            "finish_reason": finish_reason,
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        annotations = {
            "gateway": "litellm",
            "integration": "litellm-straiker",
            "source": "litellm",
            "model": model,
            "provider": provider,
            "agent_role": agent_role,
            "hook": hook,
            "litellm_hook": hook,
            "hook_tag": hook_tag,
            "trace_id": trace_id,
        }
        annotations = {k: v for k, v in annotations.items() if v is not None}
        annotations.update(ctx)

        if self.agentic:
            return {
                "source": agent_source,
                "destination": destination,
                "messages": _build_agentic_messages(messages, app_response),
                "session_id": session_id,
                "user_name": user_name,
                "user_role": user_role,
                "metadata": metadata,
                "network": network,
                "annotations": annotations,
            }
        return {
            "prompt": _last_user_prompt(messages),
            "app_response": app_response or "N/A",
            "rag_content": "N/A",
            "session_id": session_id,
            "user_name": user_name,
            "user_role": user_role,
            "metadata": metadata,
            "network": network,
            "annotations": annotations,
        }

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Straiker-Smart-Publish": "true",
        }
        if self.verbose:
            headers["Straiker-Debug"] = "TRUE"
        for k, v in self.custom_headers.items():
            if k.lower() in ("authorization", "x-straiker-smart-publish"):
                continue
            headers[k] = v
        return headers

    @staticmethod
    def _triggered_categories(debug_envelope: dict | None) -> list[str]:
        if not isinstance(debug_envelope, dict):
            return []
        detections = debug_envelope.get("detections") or {}
        block = detections.get("block") or {}
        if not isinstance(block, dict):
            return []
        return sorted(name for name, score in block.items() if isinstance(score, (int, float)) and score > 0)

    async def _call_straiker(self, payload: dict) -> tuple[float | None, str | None, str | None, DetectResponse | None]:
        try:
            body_bytes = len(json.dumps(payload, default=str).encode("utf-8"))
        except (TypeError, ValueError, OverflowError) as error:
            return None, None, f"request serialization failed: {error}", None
        if body_bytes > self.max_payload_bytes:
            return (
                None,
                None,
                (f"payload {body_bytes}B exceeds max_payload_bytes {self.max_payload_bytes}; detection unavailable"),
                None,
            )

        url = self._detect_url()
        headers = self._build_headers()
        last_err: str | None = None
        attempts = self.max_retries + 1

        for attempt in range(attempts):
            t0 = time.monotonic()
            try:
                resp = await self.async_handler.post(url, json=payload, headers=headers, timeout=self.timeout)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                if resp.status_code == 200:
                    try:
                        parsed = DetectResponse.model_validate(resp.json())
                    except (ValidationError, json.JSONDecodeError) as ve:
                        return None, None, f"invalid response schema: {ve}", None
                    return parsed.score, parsed.turn_id, None, parsed
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in RETRY_STATUS:
                    return None, None, last_err, None
            except (httpx.RequestError, asyncio.TimeoutError, Timeout) as e:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                last_err = f"{type(e).__name__}: {e}"
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                return None, None, f"{type(e).__name__}: {e}", None

            if attempt < attempts - 1:
                backoff = min(self.initial_backoff * (2**attempt), self.max_backoff)
                sleep_s = random.uniform(0, backoff)
                _structured_log(
                    logging.DEBUG,
                    "straiker.retry",
                    attempt=attempt + 1,
                    max_attempts=attempts,
                    error=last_err,
                    sleep_s=round(sleep_s, 3),
                    elapsed_ms=elapsed_ms,
                )
                await asyncio.sleep(sleep_s)

        return None, None, last_err or "unknown error", None

    def _should_dedup_pre(self, msgs: list[dict]) -> bool:
        if not (self.agentic and self.dedup_iterations):
            return False
        last = msgs[-1] if msgs else {}
        last_role = last.get("role") if isinstance(last, dict) else None
        return last_role not in ("user", "tool")

    def _messages_for_call(self, data: dict, call_type: str) -> list[dict]:
        try:
            normalized = self.get_guardrails_messages_for_call_type(call_type=CallTypes(call_type), data=data)
        except (TypeError, ValueError):
            normalized = None
        return normalized if normalized is not None else data.get("messages") or []

    def _build_block_detail(
        self,
        hook_label: str,
        score: float | None,
        turn_id: str | None,
        triggered: list[str],
        debug_envelope: dict | None,
    ) -> dict:
        detail = {
            "message": f"Straiker: threat detected ({hook_label})",
            "score": score,
            "turn_id": turn_id,
            "code": "403",
            "x-straiker-score": score,
            "x-straiker-turn-id": turn_id,
            "x-straiker-verdict": "block",
        }
        if self.verbose:
            detail["x-straiker-triggered-categories"] = triggered
            detail["straiker_debug"] = debug_envelope
        return detail

    def _raise_unavailable(self) -> None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Straiker detection unavailable",
                    "code": "503",
                    "x-straiker-verdict": "error",
                }
            },
        )

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        msgs = self._messages_for_call(data, call_type)
        if not _has_scannable_input(msgs):
            return data

        model = data.get("model")
        if self._should_dedup_pre(msgs):
            return data

        payload = self._build_payload(messages=msgs, app_response="N/A", data=data, hook="pre_call")
        t0 = time.monotonic()
        score, turn_id, err, parsed = await self._call_straiker(payload)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if err is not None:
            _structured_log(
                logging.ERROR,
                "straiker.error",
                hook="pre_call",
                error=err,
                fallback=self.unreachable_fallback,
                elapsed_ms=elapsed_ms,
            )
            if self.unreachable_fallback == "fail_open":
                return data
            self._raise_unavailable()

        verdict = "block" if (score is not None and score > self.threshold) else "allow"
        debug_envelope = getattr(parsed, "debug", None) if parsed is not None else None
        triggered = self._triggered_categories(debug_envelope)
        _structured_log(
            logging.INFO,
            "straiker.score",
            hook="pre_call",
            score=score,
            turn_id=turn_id,
            verdict=verdict,
            execution_ms=elapsed_ms,
            model=model,
            triggered_categories=triggered or None,
        )

        if verdict == "block":
            detail = self._build_block_detail("pre-call", score, turn_id, triggered, debug_envelope)
            raise HTTPException(status_code=403, detail={"error": detail})
        return data

    async def async_moderation_hook(self, data, user_api_key_dict, call_type):
        msgs = self._messages_for_call(data, call_type)
        if not _has_scannable_input(msgs):
            return data

        model = data.get("model")
        if self._should_dedup_pre(msgs):
            return data

        payload = self._build_payload(messages=msgs, app_response="N/A", data=data, hook="moderation")
        t0 = time.monotonic()
        score, turn_id, err, parsed = await self._call_straiker(payload)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if err is not None:
            _structured_log(
                logging.ERROR,
                "straiker.error",
                hook="moderation",
                error=err,
                fallback=self.unreachable_fallback,
                elapsed_ms=elapsed_ms,
            )
            if self.unreachable_fallback == "fail_open":
                return data
            self._raise_unavailable()

        verdict = "block" if (score is not None and score > self.threshold) else "allow"
        debug_envelope = getattr(parsed, "debug", None) if parsed is not None else None
        triggered = self._triggered_categories(debug_envelope)
        _structured_log(
            logging.INFO,
            "straiker.score",
            hook="moderation",
            score=score,
            turn_id=turn_id,
            verdict=verdict,
            execution_ms=elapsed_ms,
            model=model,
            triggered_categories=triggered or None,
        )

        if verdict == "block":
            detail = self._build_block_detail("during-call", score, turn_id, triggered, debug_envelope)
            raise HTTPException(status_code=403, detail={"error": detail})
        return data

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        output = _get_field(response, "output")
        is_responses_response = isinstance(output, list) and _get_field(response, "choices") is None
        raw_call_type = data.get("call_type") or data.get("litellm_call_type")
        call_type = raw_call_type if isinstance(raw_call_type, str) else "responses" if is_responses_response else ""
        msgs = list(self._messages_for_call(data, call_type))
        if not _has_scannable_input(msgs):
            return response

        model = data.get("model")
        if is_responses_response:
            response_texts, response_tool_calls = _responses_output(response)
        else:
            response_messages = [
                message
                for choice in (_get_field(response, "choices") or [])
                for message in [_get_field(choice, "message")]
                if message is not None
            ]
            response_texts = [
                text
                for message in response_messages
                for text in [_extract_text_content(_get_field(message, "content"))]
                if text
            ]
            response_tool_calls = [
                tool_call for message in response_messages for tool_call in (_get_field(message, "tool_calls") or [])
            ]
        app_response = "\n".join(response_texts)

        if not app_response and not _has_meaningful_tool_calls(response_tool_calls):
            return response

        finish_reason = "tool_calls" if _has_meaningful_tool_calls(response_tool_calls) else None
        if finish_reason is None:
            if is_responses_response:
                finish_reason = (
                    "stop"
                    if any(_get_field(item, "status") == "completed" for item in output if item is not None)
                    else None
                )
            else:
                finish_reason = next(
                    (
                        reason
                        for choice in (_get_field(response, "choices") or [])
                        for reason in [_get_field(choice, "finish_reason")]
                        if isinstance(reason, str) and reason
                    ),
                    None,
                )

        assistant_message: dict[str, Any] = {"role": "assistant"}
        if app_response:
            assistant_message["content"] = app_response
        if response_tool_calls:
            assistant_message["tool_calls"] = response_tool_calls
        msgs.append(assistant_message)

        normalized_tool_calls = _transform_tool_calls(response_tool_calls) or []
        non_agentic_response = "\n".join(
            part
            for part in (
                app_response,
                json.dumps(normalized_tool_calls, default=str) if normalized_tool_calls else "",
            )
            if part
        )
        payload = self._build_payload(
            messages=msgs,
            app_response="N/A" if self.agentic else non_agentic_response,
            data=data,
            hook="post_call",
            finish_reason=finish_reason,
        )
        t0 = time.monotonic()
        score, turn_id, err, parsed = await self._call_straiker(payload)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if err is not None:
            _structured_log(
                logging.ERROR,
                "straiker.error",
                hook="post_call",
                error=err,
                fallback=self.unreachable_fallback,
                elapsed_ms=elapsed_ms,
            )
            return response

        verdict = "block" if (score is not None and score > self.threshold) else "allow"
        debug_envelope = getattr(parsed, "debug", None) if parsed is not None else None
        triggered = self._triggered_categories(debug_envelope)
        _structured_log(
            logging.INFO,
            "straiker.score",
            hook="post_call",
            score=score,
            turn_id=turn_id,
            verdict=verdict,
            execution_ms=elapsed_ms,
            model=model,
            triggered_categories=triggered or None,
        )

        if hasattr(response, "_hidden_params") and isinstance(response._hidden_params, dict):
            hidden = {
                "score": score,
                "turn_id": turn_id,
                "verdict": verdict,
            }
            if self.verbose:
                hidden["triggered_categories"] = triggered
                hidden["straiker_debug"] = debug_envelope
            straiker_hidden = response._hidden_params.setdefault("straiker", {})
            if isinstance(straiker_hidden, dict):
                straiker_hidden.update(hidden)
        return response
