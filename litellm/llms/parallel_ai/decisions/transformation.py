"""Parallel's web-grounded Decisions use System One questions with mode-based routing.

Both LiteLLM Decisions routes share this adapter. Citations describe the whole
request, so they stay on the response rather than being assigned to answers.
Contract: https://docs.parallel.ai/api-reference/decisions/make-web-grounded-typed-decisions
"""

from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig, ir_to_systemone_request
from litellm.types.decisions import (
    DecisionsIRRequest,
    DecisionsIRResponse,
    UnsupportedDecisionsRequest,
)


class _ParallelResponse(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    mode: str
    answers: Mapping[str, Mapping[str, object]]


class ParallelAIDecisionsConfig(BaseDecisionsConfig):
    path = "/v1/decisions"
    api_key_env = ("PARALLEL_AI_API_KEY", "PARALLEL_API_KEY")
    api_base_env = ("PARALLEL_AI_API_BASE",)

    def get_default_api_base(self) -> str:
        return "https://api.parallel.ai"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, str]:
        return {
            **{
                name: value
                for name, value in headers.items()
                if name.lower() not in {"authorization", "x-api-key", "content-type"}
            },
            **({"x-api-key": api_key} if api_key is not None else {}),
            "Content-Type": "application/json",
        }

    def transform_decisions_request(
        self, model: str, request: DecisionsIRRequest, custom_llm_provider: str
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        if len(request.questions) > 100:
            return UnsupportedDecisionsRequest(reason="Parallel Decisions supports at most 100 questions")
        body: Final = ir_to_systemone_request(model, request)
        if isinstance(body, UnsupportedDecisionsRequest):
            return body
        return {"mode": model, **{key: value for key, value in body.items() if key != "model"}}

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        response: Final = _ParallelResponse.model_validate(payload)
        # Missing answers become IR refusals in the shared System One parser.
        return super().parse_response(
            {
                **response.model_dump(),
                "model": response.mode,
                "answers": {key: answer for key, answer in response.answers.items() if answer.get("type") != "refusal"},
            },
            request,
        )
