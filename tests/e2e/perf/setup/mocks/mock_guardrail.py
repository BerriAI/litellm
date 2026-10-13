"""The perf mock guardrail: a server speaking litellm's Generic Guardrail API contract
(POST /beta/litellm_basic_guardrail_api), registered once as pre_call and once as post_call.
What it does with a request is decided by the marker's flag: `-n` passes it through
(`action: NONE`), `-i` rewrites the text (`[pre]` on the request, `[post]` on the response) so the
test can see the rewrite reach the provider and the client, and `-b` blocks the request with the
reason `perf-test-block`. Its own latency comes from `additional_provider_specific_params`."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Final, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from mock_common import flag_of, marker_in

BLOCK_REASON: Final = "perf-test-block"
Action = Literal["NONE", "BLOCKED", "GUARDRAIL_INTERVENED"]


class GuardrailRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_type: Literal["request", "response"]
    texts: list[str] | None = None
    request_data: dict[str, object] = {}
    litellm_call_id: str | None = None
    model: str | None = None
    additional_provider_specific_params: dict[str, object] | None = None

    def latency_s(self) -> float:
        value: Final = (self.additional_provider_specific_params or {}).get("latency_s")
        return float(value) if isinstance(value, (int, float)) else 0.0

    def metadata(self, key: str) -> str | None:
        value: Final = self.request_data.get(key)
        return value if isinstance(value, str) else None


class GuardrailCall(BaseModel):
    marker: str | None
    input_type: Literal["request", "response"]
    action: Action
    model: str | None
    team_id: str | None
    key_alias: str | None
    user_id: str | None
    litellm_call_id: str | None
    received_at: float


class GuardrailCallsResponse(BaseModel):
    calls: list[GuardrailCall]


class GuardrailResetResponse(BaseModel):
    cleared: int


def decide(body: GuardrailRequest) -> tuple[Action, dict[str, object], str | None]:
    texts: Final = body.texts or []
    marker: Final = next((found for text in texts if (found := marker_in(text)) is not None), None)
    if marker is None:
        return "NONE", {"action": "NONE"}, None
    flag: Final = flag_of(marker)
    if flag == "b" and body.input_type == "request":
        return "BLOCKED", {"action": "BLOCKED", "blocked_reason": BLOCK_REASON}, marker
    if flag == "i":
        suffix: Final = " [pre]" if body.input_type == "request" else " [post]"
        rewritten: Final = [text + suffix if marker_in(text) else text for text in texts]
        return "GUARDRAIL_INTERVENED", {"action": "GUARDRAIL_INTERVENED", "texts": rewritten}, marker
    return "NONE", {"action": "NONE"}, marker


def build_app() -> FastAPI:
    app: Final = FastAPI(title="perf mock guardrail")
    calls: deque[GuardrailCall] = deque(maxlen=200_000)

    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def guard(body: GuardrailRequest) -> JSONResponse:
        latency: Final = body.latency_s()
        if latency > 0:
            await asyncio.sleep(latency)
        action, content, marker = decide(body)
        calls.append(
            GuardrailCall(
                marker=marker,
                input_type=body.input_type,
                action=action,
                model=body.model,
                team_id=body.metadata("user_api_key_team_id"),
                key_alias=body.metadata("user_api_key_alias"),
                user_id=body.metadata("user_api_key_user_id"),
                litellm_call_id=body.litellm_call_id,
                received_at=time.time(),
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
