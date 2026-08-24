# +-------------------------------------------------------------+
#
#           Use Conduct Guard for your LLM calls
#
#     Runtime policy enforcement — block / warn / audit / approval
#     Signed configuration + hash-chained audit + 20+ compliance packs
#     Docs: https://conductai.ai/guard
#
# +-------------------------------------------------------------+
from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import Final, Literal

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks

GUARDRAIL_NAME: Final = "conduct"


Verdict = Literal["allow", "advisory", "warning", "block", "approval", "unknown"]
FailMode = Literal["fail_open", "fail_closed"]


@dataclass(frozen=True)
class GuardDecision:
    """Structured view of what ``guard_check`` returned. The raw text is
    kept so audit / logging surfaces can quote it verbatim."""

    verdict: Verdict
    raw: str
    rule_id: str | None = None
    message: str | None = None

    @classmethod
    def parse(cls, text: str) -> GuardDecision:
        """Map the ``guard_check`` string envelope to a verdict.

        Response contract from Conduct:
          * ``"ok"`` or empty → allow silently
          * ``"advisory: ..."`` → allow but log
          * ``"WARNING — ..."`` → allow but surface
          * ``"BLOCKED — ..."`` → hard block
          * ``"PENDING approval — ..."`` → HITL — treat as block
        """
        stripped = (text or "").strip()
        if not stripped or stripped.lower().startswith("ok"):
            return cls(verdict="allow", raw=stripped)
        if stripped.startswith("BLOCKED"):
            return cls(
                verdict="block",
                raw=stripped,
                rule_id=_extract_rule_id(stripped),
                message=_strip_prefix(stripped, "BLOCKED"),
            )
        if stripped.startswith("PENDING approval"):
            return cls(
                verdict="approval",
                raw=stripped,
                rule_id=_extract_rule_id(stripped),
                message=_strip_prefix(stripped, "PENDING approval"),
            )
        if stripped.startswith("WARNING"):
            return cls(
                verdict="warning",
                raw=stripped,
                rule_id=_extract_rule_id(stripped),
                message=_strip_prefix(stripped, "WARNING"),
            )
        if stripped.startswith("advisory"):
            return cls(
                verdict="advisory",
                raw=stripped,
                rule_id=_extract_rule_id(stripped),
                message=_strip_prefix(stripped, "advisory"),
            )
        return cls(verdict="unknown", raw=stripped)


def _strip_prefix(text: str, prefix: str) -> str | None:
    remainder = text[len(prefix) :].strip()
    return remainder.lstrip(":—- ").strip() or None


def _extract_rule_id(text: str) -> str | None:
    marker = "[rule:"
    idx = text.find(marker)
    if idx < 0:
        return None
    tail = text[idx + len(marker) :]
    end = tail.find("]")
    return tail[:end].strip() if end >= 0 else None


class ConductGuardrailBlocked(Exception):
    """Raised inside the pre-call hook to abort a LiteLLM request. LiteLLM
    surfaces the message to the caller."""

    def __init__(self, decision: GuardDecision) -> None:
        self.decision = decision
        super().__init__(decision.message or decision.raw or "Blocked by Conduct Guard")


class ConductGuardrail(CustomGuardrail):
    """Conduct Guard as a LiteLLM ``CustomGuardrail``.

    Reads config from LiteLLM's guardrail block. Every pre-call hook
    invocation calls Conduct's ``guard_check`` MCP tool using the
    supplied agent token. On block, raises so the LiteLLM proxy returns
    an error to the caller instead of forwarding to the model."""

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
        ]

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        workspace_id: str | None = None,
        fail_mode: FailMode = "fail_closed",
        tool_name: str = "llm_call",
        timeout: float = 8.0,
        **kwargs: object,
    ) -> None:
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(**kwargs)

        self._api_url = (api_base or os.environ.get("CONDUCT_API_URL", "https://api.conductai.ai")).rstrip("/")
        token = api_key or os.environ.get("CONDUCT_AGENT_TOKEN")
        if not token:
            raise ValueError(
                "ConductGuardrail: agent token is required. Set CONDUCT_AGENT_TOKEN "
                "in the environment or pass api_key in the guardrail config."
            )
        self._agent_token = token
        self._workspace_id = workspace_id or os.environ.get("CONDUCT_WORKSPACE_ID")
        self._fail_mode: FailMode = fail_mode
        self._tool_name = tool_name
        self._timeout = timeout
        self._async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)

    # ── LiteLLM contract ───────────────────────────────────────────────

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: object,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> dict | None:
        decision = await self._check(data=data, call_type=call_type)

        if decision.verdict in ("block", "approval"):
            raise ConductGuardrailBlocked(decision)

        data.setdefault("metadata", {}).setdefault("conduct_guard", {}).update(
            {"verdict": decision.verdict, "rule_id": decision.rule_id}
        )
        return data

    # ── Guard check ─────────────────────────────────────────────────

    async def _check(self, *, data: dict, call_type: str) -> GuardDecision:
        tool_input = _build_tool_input(data, call_type)
        session_id = _extract_session_id(data)
        prompt = _extract_prompt_text(data)

        arguments: dict = {
            "tool_name": self._tool_name,
            "tool_input": tool_input,
        }
        if prompt is not None:
            arguments["prompt"] = prompt

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": "guard_check", "arguments": arguments},
        }

        headers = {
            "Authorization": f"Bearer {self._agent_token}",
            "Content-Type": "application/json",
            "User-Agent": "litellm-conduct-guardrail/1.0",
            "X-Claude-Surface": "litellm",
        }
        if self._workspace_id:
            headers["X-Workspace-Id"] = self._workspace_id
        if session_id:
            headers["X-Conduct-Session-Id"] = session_id

        try:
            response = await self._async_handler.post(
                f"{self._api_url}/guard/mcp",
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()

            if "error" in body:
                err = body["error"]
                verbose_proxy_logger.warning("conduct_guard: eval error %s — applying %s", err, self._fail_mode)
                if self._fail_mode == "fail_closed":
                    return GuardDecision(
                        verdict="block",
                        raw=str(err),
                        message="Conduct Guard policy-eval error (fail_closed).",
                    )
                return GuardDecision(verdict="allow", raw="fail_open")

            result = body.get("result") or {}
            for item in result.get("content", []) or []:
                if item.get("type") == "text":
                    return GuardDecision.parse(item.get("text", ""))
            return GuardDecision(verdict="allow", raw="")
        except Exception as e:  # noqa: BLE001 — transport failure fallback path is intentionally broad
            verbose_proxy_logger.warning("conduct_guard: transport error %s — applying %s", e, self._fail_mode)
            if self._fail_mode == "fail_closed":
                return GuardDecision(
                    verdict="block",
                    raw=str(e),
                    message="Conduct Guard is unreachable (fail_closed).",
                )
            return GuardDecision(verdict="allow", raw="fail_open")


# ── Helpers ─────────────────────────────────────────────────────────


def _extract_session_id(data: dict) -> str | None:
    metadata = data.get("litellm_metadata") or data.get("metadata") or {}
    for key in ("trace_id", "X-Conduct-Session-Id", "conduct_session_id"):
        val = metadata.get(key)
        if val:
            return str(val)

    user = data.get("user") or metadata.get("user") or ""
    first_msg = ""
    for m in data.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "user":
            first_msg = str(m.get("content", ""))[:512]
            break
    if not user and not first_msg:
        return None
    digest = hashlib.sha256((user + "|" + first_msg).encode("utf-8")).hexdigest()
    return f"litellm-{digest[:16]}"


def _extract_prompt_text(data: dict) -> str | None:
    for m in reversed(data.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content[:4000]
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                return " ".join(parts)[:4000] or None
    return None


def _build_tool_input(data: dict, call_type: str) -> dict:
    messages = data.get("messages") or []
    return {
        "model": data.get("model"),
        "call_type": call_type,
        "message_count": len(messages),
        "temperature": data.get("temperature"),
        "max_tokens": data.get("max_tokens"),
        "stream": bool(data.get("stream")),
        "content": _extract_prompt_text(data) or "",
    }
