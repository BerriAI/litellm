from collections.abc import Mapping
from typing import Final

import httpx
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
    openai_evidence_fields = frozenset({"citations", "decision_id", "mode"})

    def get_default_api_base(self) -> str:
        return "https://api.parallel.ai"

    def resolve_connection(self, api_base: str | None, api_key: str | None) -> tuple[str | None, str | None]:
        configured_base: Final = self.resolve_api_base(None)
        resolved_base: Final = api_base or configured_base
        destination: Final = self._decisions_url(resolved_base)
        if api_key:
            return resolved_base, api_key
        if destination not in {self._decisions_url(None), self._decisions_url(configured_base)}:
            raise ValueError(
                "Parallel Decisions requires an explicit api_key for a custom api_base unless "
                "the destination is configured in PARALLEL_AI_API_BASE"
            )
        return resolved_base, self.resolve_api_key(None)

    def get_complete_url(self, api_base: str, model: str) -> str:
        return super().get_complete_url(api_base.rstrip("/").removesuffix("/v1/search").removesuffix(self.path), model)

    def _decisions_url(self, api_base: str | None) -> httpx.URL:
        return httpx.URL(self.get_complete_url(api_base or self.get_default_api_base(), "decision"))

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
        if model != "decision":
            return UnsupportedDecisionsRequest(reason="Parallel Decisions supports the decision model")
        if len(request.questions) > 100:
            return UnsupportedDecisionsRequest(reason="Parallel Decisions supports at most 100 questions")
        body: Final = ir_to_systemone_request(model, request)
        if isinstance(body, UnsupportedDecisionsRequest):
            return body
        return {"mode": "fast", **{key: value for key, value in body.items() if key != "model"}}

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        response: Final = _ParallelResponse.model_validate(payload)
        return super().parse_response(
            {
                **response.model_dump(),
                "model": "decision",
                "answers": {key: answer for key, answer in response.answers.items() if answer.get("type") != "refusal"},
            },
            request,
        )
