"""The perf mock guardrail: a server speaking litellm's Generic Guardrail API contract
(POST /beta/litellm_basic_guardrail_api), registered once as pre_call and once as post_call.
What it does with a request is decided by the marker's flag: `-n` passes it through
(`action: NONE`), `-i` rewrites the text (`[pre]` on the request, `[post]` on the response) so the
test can see the rewrite reach the provider and the client, and `-b` blocks the request with the
reason `perf-test-block`. Its own latency comes from `additional_provider_specific_params`."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from typing import Final, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mock_common import flag_of, marker_in
from pydantic import BaseModel, ConfigDict

BLOCK_REASON: Final = "perf-test-block"
Action = Literal["NONE", "BLOCKED", "GUARDRAIL_INTERVENED"]


class GuardrailRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_type: Literal["request", "response"]
    texts: list[str] | None = None
    litellm_call_id: str | None = None
    additional_provider_specific_params: dict[str, object] | None = None

    def latency_s(self) -> float:
        value: Final = (self.additional_provider_specific_params or {}).get("latency_s")
        return float(value) if isinstance(value, (int, float)) else 0.0


class GuardrailCall(BaseModel):
    marker: str | None
    input_type: Literal["request", "response"]
    action: Action


class GuardrailCallsResponse(BaseModel):
    calls: list[GuardrailCall]


class GuardrailResetResponse(BaseModel):
    cleared: int


def marker_of(body: GuardrailRequest, remembered: Mapping[str, str]) -> str | None:
    """The request's marker: in the prompt on the request hook, remembered by `litellm_call_id` on the response
    hook, since a generated provider reply does not contain it."""
    texts: Final = body.texts or []
    in_text: Final = next((found for text in texts if (found := marker_in(text)) is not None), None)
    if in_text is not None or body.litellm_call_id is None:
        return in_text
    return remembered.get(body.litellm_call_id)


def decide(body: GuardrailRequest, marker: str | None) -> tuple[Action, dict[str, object]]:
    texts: Final = body.texts or []
    if marker is None:
        return "NONE", {"action": "NONE"}
    flag: Final = flag_of(marker)
    if flag == "b" and body.input_type == "request":
        return "BLOCKED", {"action": "BLOCKED", "blocked_reason": BLOCK_REASON}
    if flag == "i":
        suffix: Final = " [pre]" if body.input_type == "request" else " [post]"
        rewritten: Final = (
            [text + suffix if marker_in(text) else text for text in texts]
            if body.input_type == "request"
            else [text + suffix for text in texts]
        )
        return "GUARDRAIL_INTERVENED", {"action": "GUARDRAIL_INTERVENED", "texts": rewritten}
    return "NONE", {"action": "NONE"}


def build_app() -> FastAPI:
    app: Final = FastAPI(title="perf mock guardrail")
    calls: deque[GuardrailCall] = deque(maxlen=200_000)
    markers: dict[str, str] = {}  # mutable-ok: the request hook's marker, looked up by the response hook

    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def guard(body: GuardrailRequest) -> JSONResponse:
        latency: Final = body.latency_s()
        if latency > 0:
            await asyncio.sleep(latency)
        marker: Final = marker_of(body, markers)
        if marker is not None and body.litellm_call_id is not None and body.input_type == "request":
            markers[body.litellm_call_id] = marker
        action, content = decide(body, marker)
        calls.append(
            GuardrailCall(
                marker=marker,
                input_type=body.input_type,
                action=action,
            )
        )
        return JSONResponse(content=content)

    async def calls_seen() -> GuardrailCallsResponse:
        return GuardrailCallsResponse(calls=list(calls))

    async def reset() -> GuardrailResetResponse:
        count: Final = len(calls)
        calls.clear()
        return GuardrailResetResponse(cleared=count)

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/beta/litellm_basic_guardrail_api", guard, methods=["POST"])
    app.add_api_route("/mock/calls", calls_seen, methods=["GET"])
    app.add_api_route("/mock/reset", reset, methods=["POST"])
    return app
