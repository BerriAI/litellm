"""
Transform request/response for DashScope multimodal embeddings.
Reference: https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference
"""

from typing import List, Optional, Union
import httpx
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage
from ..common_utils import DashScopeError

DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"


class DashScopeMultimodalEmbeddingConfig(BaseEmbeddingConfig):

    @staticmethod
    def is_multimodal_embedding(model: str) -> bool:
        """Return True if the model name indicates a multimodal embedding model.

        A model is considered multimodal when its name (after stripping the
        provider prefix) contains any of the following signals:
          - "multimodal"  (e.g. multimodal-embedding-v1)
          - "vl"          (e.g. qwen3-vl-embedding, qwen2.5-vl-embedding)
          - "vision"      (e.g. tongyi-embedding-vision-flash)

        This avoids maintaining a hardcoded allowlist: any future DashScope
        multimodal embedding model that follows the naming convention will be
        routed correctly without a code change.
        """
        base = model.split("/")[-1].lower() if "/" in model else model.lower()
        return any(kw in base for kw in ("multimodal", "vl", "vision"))

    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["dimensions"]

    def map_openai_params(self, non_default_params: dict, optional_params: dict,
                          model: str, drop_params: bool = False) -> dict:
        if "dimensions" in non_default_params:
            optional_params["dimension"] = non_default_params["dimensions"]
        return optional_params

    def validate_environment(self, headers: dict, model: str,
                             messages: List[AllMessageValues],
                             optional_params: dict, litellm_params: dict,
                             api_key: Optional[str] = None,
                             api_base: Optional[str] = None) -> dict:
        if api_key is None:
            api_key = get_secret_str("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError("DashScope API key is required. "
                             "Set 'DASHSCOPE_API_KEY' env var or pass api_key explicitly.")
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}", **headers}

    def get_complete_url(self, api_base: Optional[str], api_key: Optional[str],
                         model: str, optional_params: dict, litellm_params: dict,
                         stream: Optional[bool] = None) -> str:
        user_base = get_secret_str("DASHSCOPE_API_BASE")
        if user_base:
            return user_base
        if api_base and "/compatible-mode/" in api_base:
            return DEFAULT_API_BASE
        if api_base:
            base = api_base.rstrip("/")
            if base.endswith("/multimodal-embedding/multimodal-embedding"):
                return base
            if base.endswith("/multimodal-embedding"):
                return base
            return f"{base}/services/embeddings/multimodal-embedding/multimodal-embedding"
        return DEFAULT_API_BASE

    def _normalize_content_blocks(self, content: list) -> list:
        """Convert OpenAI content blocks to DashScope native format."""
        result = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                result.append({"text": block.get("text", "")})
            elif block_type == "image_url":
                image_url = block.get("image_url", {})
                if isinstance(image_url, dict):
                    image_url = image_url.get("url", "")
                result.append({"image": image_url})
            else:
                result.append(block)
        return result

    def _normalize_input_item(self, item: Union[str, list, dict]) -> dict:
        if isinstance(item, str):
            return {"text": item}
        if isinstance(item, list):
            blocks = self._normalize_content_blocks(item)
            if len(blocks) == 1:
                return blocks[0]
            return {"content_list": blocks}
        return item

    def transform_embedding_request(self, model: str,
                                    input: AllEmbeddingInputValues,
                                    optional_params: dict,
                                    headers: dict) -> dict:
        inputs = input if isinstance(input, list) else [input]
        contents = [self._normalize_input_item(item) for item in inputs]
        body: dict = {"model": model, "input": {"contents": contents}}
        params = {}
        if "dimension" in optional_params:
            params["dimension"] = optional_params["dimension"]
        if "output_type" in optional_params:
            params["output_type"] = optional_params["output_type"]
        if params:
            body["parameters"] = params
        return body

    def transform_embedding_response(self, model: str,
                                     raw_response: httpx.Response,
                                     model_response: EmbeddingResponse,
                                     logging_obj: LiteLLMLoggingObj,
                                     api_key: Optional[str],
                                     request_data: dict,
                                     optional_params: dict,
                                     litellm_params: dict) -> EmbeddingResponse:
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise DashScopeError(status_code=raw_response.status_code,
                                 message=f"Failed to parse DashScope response as JSON: {str(e)}")
        logging_obj.post_call(input=request_data.get("input"), api_key=api_key,
                              additional_args={"complete_input_dict": request_data},
                              original_response=response_json)
        if "code" in response_json:
            raise DashScopeError(status_code=raw_response.status_code,
                                 message=response_json.get("message", str(response_json)))
        output = response_json.get("output", {})
        embeddings = output.get("embeddings", [])
        model_response.object = "list"
        model_response.data = [
            {"object": "embedding", "index": emb.get("index", i),
             "embedding": emb.get("embedding", [])}
            for i, emb in enumerate(embeddings)
        ]
        model_response.model = model
        usage = response_json.get("usage") or {}
        input_tokens = usage.get("input_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens)

        text_tokens = None
        image_tokens = None
        if "input_tokens_details" in usage:
            image_tokens = usage["input_tokens_details"].get("image_tokens")
            text_tokens = usage["input_tokens_details"].get("text_tokens")
        elif "image_tokens" in usage:
            image_tokens = usage["image_tokens"]
            text_tokens = input_tokens
            input_tokens = input_tokens + image_tokens
            total_tokens = max(total_tokens, input_tokens)

        prompt_tokens_details = None
        if image_tokens is not None or text_tokens is not None:
            from litellm.types.utils import PromptTokensDetailsWrapper
            prompt_tokens_details = PromptTokensDetailsWrapper(
                image_tokens=image_tokens,
                text_tokens=text_tokens,
            )

        setattr(model_response, "usage",
                Usage(prompt_tokens=input_tokens, completion_tokens=0,
                      total_tokens=total_tokens,
                      prompt_tokens_details=prompt_tokens_details))
        if "request_id" in response_json:
            setattr(model_response, "id", response_json["request_id"])
        return model_response

    def get_error_class(self, error_message: str, status_code: int,
                        headers: Union[dict, httpx.Headers]) -> BaseLLMException:
        if isinstance(headers, dict):
            headers = httpx.Headers(headers)
        return DashScopeError(status_code=status_code, message=error_message, headers=headers)
