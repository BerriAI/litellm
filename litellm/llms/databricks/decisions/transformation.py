from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.types.decisions import DecisionsIRRequest, UnsupportedDecisionsRequest

DATABRICKS_AI_DECIDE_MODEL: Final = "ai_decide"
_MAPPING_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


def validate_ai_decide_model(model: str) -> str:
    if model != DATABRICKS_AI_DECIDE_MODEL:
        raise ValueError(
            f"Databricks decisions model must be {DATABRICKS_AI_DECIDE_MODEL!r} (the ai_decide AI Function)"
        )
    return model


def databricks_workspace_host(api_base: str) -> str:
    return api_base.rstrip("/").removesuffix("/serving-endpoints")


def _mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _systemone_answer(answer: object) -> object:
    fields: Final = _mapping(answer)
    if fields is None or fields.get("type") != "noul" or "probability" not in fields:
        return answer
    return MappingProxyType(
        {**{key: value for key, value in fields.items() if key != "probability"}, "noul": fields["probability"]}
    )


def ai_decide_response(payload: Mapping[str, object]) -> Mapping[str, object]:
    response: Final = _mapping(payload.get("response"))
    answers: Final = None if response is None else _mapping(response.get("answers"))
    if response is None or answers is None:
        return payload
    return MappingProxyType(
        {
            **{key: value for key, value in payload.items() if key != "response"},
            **response,
            "answers": MappingProxyType({key: _systemone_answer(answer) for key, answer in answers.items()}),
        }
    )


@dataclass(frozen=True, slots=True)
class DatabricksDecisionsConnection:
    api_base: str
    api_key: str = field(repr=False)


class DatabricksDecisionsConfig(BaseDecisionsConfig):
    path = "/api/2.0/ai-functions/ai-decide"
    api_key_env = ("DATABRICKS_API_KEY", "DATABRICKS_TOKEN")
    api_base_env = ("DATABRICKS_API_BASE",)

    def missing_api_base_message(self, custom_llm_provider: str) -> str:
        return (
            f"api_base is required for Decisions provider '{custom_llm_provider}': set DATABRICKS_API_BASE to "
            "https://<workspace-host>"
        )

    def canonical_model(self, model: str) -> str:
        return validate_ai_decide_model(model)

    def get_complete_url(self, api_base: str, model: str) -> str:
        return f"{databricks_workspace_host(api_base)}{self.path}"

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        body: Final = super().transform_decisions_request(model, request, custom_llm_provider)
        if isinstance(body, UnsupportedDecisionsRequest):
            return body
        return MappingProxyType({key: value for key, value in body.items() if key != "model"})

    def unwrap_response(self, payload: object) -> object:
        fields: Final = _mapping(payload)
        return payload if fields is None else ai_decide_response(fields)

    def classifier_response(self, body: Mapping[str, object], requested_model: str) -> Mapping[str, object]:
        return MappingProxyType({**ai_decide_response(body), "model": requested_model})

    def connection(self, api_base: str | None, api_key: str | None) -> DatabricksDecisionsConnection:
        base: Final = self.resolve_api_base(api_base)
        key: Final = api_key if api_base is not None else self.resolve_api_key(api_key)
        if not base or not key:
            raise ValueError(
                "Databricks requires api_key or DATABRICKS_API_KEY (or DATABRICKS_TOKEN) and api_base or "
                "DATABRICKS_API_BASE pointing to https://<workspace-host>"
            )
        return DatabricksDecisionsConnection(api_base=databricks_workspace_host(base), api_key=key)


DATABRICKS_DECISIONS_CONFIG: Final[DatabricksDecisionsConfig] = DatabricksDecisionsConfig()
