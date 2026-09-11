"""DashScope native text reranking with input/parameters and output.results envelopes.

qwen3.7-text-rerank was verified in Beijing, including return_documents=true.
Protocol routing through brand aliases does not establish regional model availability.
Docs: https://help.aliyun.com/zh/model-studio/text-rerank-api
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit, urlunsplit

from pydantic import TypeAdapter

from .transformation import DashScopeRerankConfig, DashScopeRerankUsage


class DashScopeNativeRerankConfig(DashScopeRerankConfig):
    def __init__(self, provider_config: DashScopeRerankConfig, api_base: str | None = None) -> None:
        self._provider_config: Final = provider_config
        self._api_base: Final = api_base

    def _resolve_api_key(self, api_key: str | None) -> str:
        return self._provider_config._resolve_api_key(api_key)

    def _resolve_rerank_api_base(self, api_base: str | None) -> str:
        return self._provider_config._resolve_rerank_api_base(api_base)

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: Mapping[str, object] | None = None,
    ) -> str:
        native_base: Final = self._api_base or self._resolve_rerank_api_base(api_base)
        parsed: Final = urlsplit(native_base.rstrip("/"))
        if parsed.path.endswith("/services/rerank/text-rerank/text-rerank"):
            return urlunsplit(parsed)
        native_path: Final = parsed.path.removesuffix("/compatible-mode/v1").removesuffix("/compatible-api/v1/reranks")
        api_path: Final = native_path if native_path.endswith("/api/v1") else f"{native_path}/api/v1"
        return urlunsplit(parsed._replace(path=f"{api_path}/services/rerank/text-rerank/text-rerank"))

    def get_supported_cohere_rerank_params(self, model: str) -> list[str]:
        return [*super().get_supported_cohere_rerank_params(model), "instruction"]

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Mapping[str, object],
        headers: Mapping[str, object],
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        request: Final = super().transform_rerank_request(model, optional_rerank_params, headers, litellm_params)
        return {
            "model": model,
            "input": {"query": request["query"], "documents": request["documents"]},
            "parameters": {
                ("instruct" if name == "instruction" else name): optional_rerank_params[name]
                for name in ("top_n", "return_documents", "instruction")
                if optional_rerank_params.get(name) is not None
            },
        }

    def _get_request_query(self, request_data: Mapping[str, object]) -> object:
        return (
            TypeAdapter(Mapping[str, object])
            .validate_python(request_data.get("input", MappingProxyType({})))
            .get("query")
        )

    def _get_response_fields(
        self, response_json: Mapping[str, object], usage: DashScopeRerankUsage
    ) -> tuple[object, object, int | None]:
        output: Final = TypeAdapter(Mapping[str, object]).validate_python(
            response_json.get("output", MappingProxyType({}))
        )
        return output.get("results"), response_json.get("request_id"), usage.get("prompt_tokens")
