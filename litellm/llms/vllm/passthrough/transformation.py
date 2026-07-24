from typing import TYPE_CHECKING, Optional, Tuple

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig

from ..common_utils import VLLMModelInfo

if TYPE_CHECKING:
    from httpx import URL, Response

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import CostResponseTypes


class VLLMPassthroughConfig(VLLMModelInfo, BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        return "stream" in request_data

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        endpoint: str,
        request_query_params: Optional[dict],
        litellm_params: dict,
    ) -> Tuple["URL", str]:
        base_target_url = self.get_api_base(api_base)

        if base_target_url is None:
            raise Exception("VLLM api base not found")

        return (
            self.format_url(endpoint, base_target_url, request_query_params),
            base_target_url,
        )

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: "Response",
        request_data: dict,
        logging_obj: "LiteLLMLoggingObj",
        endpoint: str,
    ) -> Optional["CostResponseTypes"]:
        from litellm import encoding
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
        from litellm.types.utils import EmbeddingResponse, ModelResponse, Usage

        if "chat/completions" in endpoint:
            return OpenAIGPTConfig().transform_response(
                model=model,
                messages=[{"role": "user", "content": "no-message-pass-through-endpoint"}],
                raw_response=httpx_response,
                model_response=ModelResponse(),
                logging_obj=logging_obj,
                optional_params={},
                litellm_params={},
                api_key="",
                request_data=request_data,
                encoding=encoding,
            )

        endpoint_name = endpoint.rstrip("/").rsplit("/", 1)[-1]
        if endpoint_name not in {"pooling", "embeddings", "classify", "score", "rerank"}:
            return None

        try:
            response_json = httpx_response.json()
        except ValueError:
            return None

        if not isinstance(response_json, dict):
            return None

        usage = response_json.get("usage")
        if not isinstance(usage, dict):
            return None

        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        return EmbeddingResponse(
            model=response_json.get("model") or model,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                total_tokens=usage.get("total_tokens") or prompt_tokens,
            ),
        )
