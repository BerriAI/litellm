from litellm.llms.hosted_vllm.rerank.transformation import HostedVLLMRerankConfig
from litellm.secret_managers.main import get_secret_str


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
