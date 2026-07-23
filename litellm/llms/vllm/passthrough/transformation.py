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
        """
        Build a recognized litellm response from a non-streaming vLLM passthrough
        response so ``_success_handler_helper_fn`` can construct a
        ``standard_logging_object``. Without it, router-model passthrough
        requests (``/vllm/*`` resolving to a ``model_list`` deployment) never
        create a ``LiteLLM_SpendLogs`` row and every call raises in the cost and
        Prometheus callbacks
        """
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

        try:
            response_body = httpx_response.json()
        except ValueError:
            return None

        if not isinstance(response_body, dict):
            return None

        usage = response_body.get("usage")
        if not isinstance(usage, dict):
            return None

        return EmbeddingResponse(
            model=response_body.get("model") or model,
            usage=Usage(**usage),
        )
