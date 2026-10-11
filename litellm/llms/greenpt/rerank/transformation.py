from collections.abc import Mapping, Sized
from typing import Final

from litellm.llms.hosted_vllm.rerank.transformation import HostedVLLMRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import RerankBilledUnits
from litellm.types.utils import ModelInfo


class GreenPTRerankConfig(HostedVLLMRerankConfig):
    def get_supported_cohere_rerank_params(self, model: str) -> list[str]:  # mutable-ok: mirrors override contract
        return ["query", "documents", "top_n", "return_documents"]

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: mirrors override contract
        model: str,
        api_key: str | None = None,
        optional_params: dict[str, object] | None = None,  # mutable-ok: mirrors override contract
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict[str, str]:  # mutable-ok: mirrors override contract
        resolved_api_key: Final = api_key or get_secret_str("GREENPT_API_KEY")
        if resolved_api_key is None:
            raise ValueError("GreenPT API key is required. Set GREENPT_API_KEY.")
        return {
            "Authorization": f"Bearer {resolved_api_key}",
            "accept": "application/json",
            "content-type": "application/json",
            **headers,
        }

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: dict[str, object],  # mutable-ok: mirrors override contract
        headers: dict[str, str],  # mutable-ok: mirrors override contract
        litellm_params: dict[str, object] | None = None,  # mutable-ok: mirrors override contract
    ) -> dict[str, object]:  # mutable-ok: mirrors override contract
        request: Final[Mapping[str, object]] = super().transform_rerank_request(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # HostedVLLMRerankConfig returns an untyped dict
            model=model,
            optional_rerank_params=optional_rerank_params,
            headers=headers,
            litellm_params=litellm_params,
        )
        documents: Final = optional_rerank_params["documents"]
        top_n_default: Final = len(documents) if isinstance(documents, Sized) else None
        return {"top_n": top_n_default, **request} if top_n_default is not None else {**request}

    def calculate_rerank_cost(
        self,
        model: str,
        custom_llm_provider: str | None = None,
        billed_units: RerankBilledUnits | None = None,
        model_info: ModelInfo | None = None,
    ) -> tuple[float, float]:
        if model_info is None or billed_units is None:
            return 0.0, 0.0
        input_cost_per_token: Final = model_info.get("input_cost_per_token")
        if input_cost_per_token is None:
            return 0.0, 0.0
        total_tokens: Final = billed_units.get("total_tokens")
        if total_tokens is None:
            return 0.0, 0.0
        return input_cost_per_token * total_tokens, 0.0
