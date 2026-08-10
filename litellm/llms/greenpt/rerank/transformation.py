from litellm.llms.hosted_vllm.rerank.transformation import HostedVLLMRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import RerankBilledUnits
from litellm.types.utils import ModelInfo


class GreenPTRerankConfig(HostedVLLMRerankConfig):
    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return ["query", "documents", "top_n", "return_documents"]

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        optional_params: dict | None = None,
    ) -> dict:
        api_key = api_key or get_secret_str("GREENPT_API_KEY")
        if api_key is None:
            raise ValueError("GreenPT API key is required. Set GREENPT_API_KEY.")
        return super().validate_environment(
            headers=headers,
            model=model,
            api_key=api_key,
            optional_params=optional_params,
        )

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: dict,
        headers: dict,
        litellm_params: dict | None = None,
    ) -> dict:
        request = super().transform_rerank_request(
            model=model,
            optional_rerank_params=optional_rerank_params,
            headers=headers,
            litellm_params=litellm_params,
        )
        request.setdefault("top_n", len(request["documents"]))
        return request

    def calculate_rerank_cost(
        self,
        model: str,
        custom_llm_provider: str | None = None,
        billed_units: RerankBilledUnits | None = None,
        model_info: ModelInfo | None = None,
    ) -> tuple[float, float]:
        if model_info is None or billed_units is None:
            return 0.0, 0.0
        input_cost_per_token = model_info.get("input_cost_per_token")
        if input_cost_per_token is None:
            return 0.0, 0.0
        total_tokens = billed_units.get("total_tokens")
        if total_tokens is None:
            return 0.0, 0.0
        return input_cost_per_token * total_tokens, 0.0
