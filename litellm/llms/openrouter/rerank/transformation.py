"""
OpenRouter Rerank API Configuration.

OpenRouter serves rerank models (e.g. `cohere/rerank-v3.5`) through
`POST https://openrouter.ai/api/v1/rerank`. The request and response shapes
follow the Cohere rerank format, with OpenRouter-specific response fields
(`provider`, and a `usage` block carrying `search_units` / `total_tokens`).

Reference: https://openrouter.ai/docs/api/api-reference/rerank/submit-a-rerank-request
"""

from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankResponseResult,
)

from ..common_utils import OpenRouterException

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_OBJECTS: Final = TypeAdapter(list[Mapping[str, object]], config=ConfigDict(hide_input_in_errors=True))
_INDEX: Final = TypeAdapter(int, config=ConfigDict(strict=True, hide_input_in_errors=True))
_SCORE: Final[TypeAdapter[bool | int | float]] = TypeAdapter(
    bool | int | float, config=ConfigDict(strict=True, hide_input_in_errors=True)
)
_BILLED_UNITS: Final = TypeAdapter(RerankBilledUnits)
_STR: Final = TypeAdapter(str)

DEFAULT_API_BASE: Final = "https://openrouter.ai/api/v1"


class OpenRouterRerankConfig(BaseRerankConfig):
    """
    Reference: https://openrouter.ai/docs/api/api-reference/rerank/submit-a-rerank-request
    """

    def get_supported_cohere_rerank_params(self, model: str) -> list[str]:  # mutable-ok: BaseRerankConfig contract
        return [
            "query",
            "documents",
            "top_n",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: dict[str, object] | None,  # mutable-ok: BaseRerankConfig contract
        model: str,
        drop_params: bool,
        query: str,
        documents: list[str | dict[str, object]],  # mutable-ok: BaseRerankConfig contract
        custom_llm_provider: str | None = None,
        top_n: int | None = None,
        rank_fields: list[str] | None = None,  # mutable-ok: BaseRerankConfig contract
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        max_tokens_per_doc: int | None = None,
        instruction: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: BaseRerankConfig contract
        if non_default_params is None:
            return {}
        supported_params: Final = self.get_supported_cohere_rerank_params(model)
        return {k: v for k, v in non_default_params.items() if k in supported_params}

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: Mapping[str, object] | None = None,
    ) -> str:
        base: Final = api_base or get_secret_str("OPENROUTER_API_BASE") or DEFAULT_API_BASE
        return join_api_base(base, "rerank")

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseRerankConfig contract
        key: Final = api_key or get_secret_str("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OpenRouter API key not found. Pass `api_key=...` or set the OPENROUTER_API_KEY environment variable."
            )
        provider_headers: Final = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": get_secret_str("OR_SITE_URL") or "https://litellm.ai",
            "X-Title": get_secret_str("OR_APP_NAME") or "liteLLM",
        }
        caller_headers: Final = {name: value for name, value in headers.items() if name != "Authorization"}
        return {**caller_headers, **provider_headers}

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: dict[str, object],  # mutable-ok: BaseRerankConfig contract
        headers: Mapping[str, str],
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict[str, object]:  # mutable-ok: BaseRerankConfig contract
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for OpenRouter rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for OpenRouter rerank")

        # OpenRouter expects the upstream model id (e.g. `cohere/rerank-v3.5`),
        # without LiteLLM's `openrouter/` routing prefix.
        upstream_model: Final = model.removeprefix("openrouter/")
        return {"model": upstream_model, **optional_rerank_params}

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: BaseRerankConfig contract
        optional_params: dict[str, object] | None = None,  # mutable-ok: BaseRerankConfig contract
        litellm_params: dict[str, object] | None = None,  # mutable-ok: BaseRerankConfig contract
    ) -> RerankResponse:
        if raw_response.status_code != 200:
            raise OpenRouterException(message=raw_response.text, status_code=raw_response.status_code)

        logging_obj.post_call(original_response=raw_response.text)  # pyright: ignore[reportUnknownMemberType]  # Logging.post_call is untyped upstream

        _json_response: Final = _JSON_OBJECT.validate_python(raw_response.json())

        _results: Final = _json_response.get("results")
        if _results is None:
            raise ValueError(f"No results found in the response={_json_response}")

        # OpenRouter returns Cohere-shaped results:
        # {"index": 0, "relevance_score": 0.98, "document": {"text": "..."}}
        transformed_results: Final = [
            self._transform_result(result) for result in _JSON_OBJECTS.validate_python(_results)
        ]

        usage: Final = _JSON_OBJECT.validate_python(_json_response.get("usage", {}))
        billed_units: Final = _BILLED_UNITS.validate_python(usage)
        meta: Final[RerankResponseMeta] = {"billed_units": billed_units}

        rerank_response: Final = RerankResponse(
            id=_STR.validate_python(_json_response.get("id") or str(uuid.uuid4())),
            results=transformed_results,
            meta=meta,
        )

        # Keep OpenRouter-specific fields for logging/cost tracking
        provider_name: Final = _json_response.get("provider")
        if isinstance(provider_name, str):
            rerank_response.hidden_params["provider"] = provider_name
        rerank_response.hidden_params["model"] = model

        return rerank_response

    @staticmethod
    def _transform_result(result: Mapping[str, object]) -> RerankResponseResult:
        transformed_result: Final[RerankResponseResult] = {
            "index": _INDEX.validate_python(result["index"]),
            "relevance_score": float(_SCORE.validate_python(result["relevance_score"])),
        }
        document: Final = result.get("document")
        if isinstance(document, str):
            transformed_result["document"] = {"text": document}
        elif isinstance(document, Mapping):
            document_mapping: Final = _JSON_OBJECT.validate_python(document)
            text: Final = document_mapping.get("text")
            if isinstance(text, str):
                transformed_result["document"] = {"text": text}
        return transformed_result

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,  # mutable-ok: BaseRerankConfig contract
    ) -> OpenRouterException:
        return OpenRouterException(
            status_code=status_code,
            message=self._extract_error_message(error_message),
            headers=headers,
        )

    @staticmethod
    def _extract_error_message(error_message: str) -> str:
        # OpenRouter errors are OpenAI-shaped: {"error": {"message": "...", "code": ...}}
        try:
            parsed: Final = _JSON_OBJECT.validate_json(error_message)
        except Exception:
            return error_message
        error_obj: Final = parsed.get("error")
        if isinstance(error_obj, Mapping):
            error_mapping: Final = _JSON_OBJECT.validate_python(error_obj)
            inner: Final = error_mapping.get("message")
            if isinstance(inner, str):
                return inner
        return error_message
