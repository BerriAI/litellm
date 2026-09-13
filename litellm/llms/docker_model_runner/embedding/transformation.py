"""
Docker Model Runner Embedding API Configuration.

Docker Model Runner provides OpenAI-compatible embeddings via:
/engines/v1/embeddings

Docs: https://docs.docker.com/ai/model-runner/api-reference/
"""

from typing import Final

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse
from litellm.utils import convert_to_model_response_object


class DockerModelRunnerEmbeddingError(BaseLLMException):
    pass


class DockerModelRunnerEmbeddingConfig(BaseEmbeddingConfig):
    def validate_environment(
        self,
        headers: dict,  # mutable-ok: signature dictated by BaseEmbeddingConfig
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: signature dictated by BaseEmbeddingConfig
        optional_params: dict,  # mutable-ok: signature dictated by BaseEmbeddingConfig
        litellm_params: dict,  # mutable-ok: signature dictated by BaseEmbeddingConfig
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: signature dictated by BaseEmbeddingConfig
        default_headers: Final = {  # mutable-ok: API request payload
            "Content-Type": "application/json",
        }

        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        else:
            default_headers["Authorization"] = "Bearer dummy-key"

        return {**default_headers, **headers}  # mutable-ok: API request payload

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: signature dictated by BaseEmbeddingConfig
        litellm_params: dict,  # mutable-ok: signature dictated by BaseEmbeddingConfig
        stream: bool | None = None,
    ) -> str:
        if api_base is None:
            api_base = get_secret_str(  # rebind-ok: normalize the argument locally, mirrors hosted_vllm
                "DOCKER_MODEL_RUNNER_API_BASE"
            )
            if api_base is None:
                api_base = "http://localhost:12434/engines/v1"  # rebind-ok: normalize the argument locally, mirrors hosted_vllm

        api_base = api_base.rstrip("/")  # rebind-ok: normalize the argument locally, mirrors hosted_vllm

        if not api_base.endswith("/embeddings"):
            api_base = f"{api_base}/embeddings"  # rebind-ok: normalize the argument locally, mirrors hosted_vllm

        return api_base

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,  # mutable-ok: signature dictated by BaseEmbeddingConfig
        headers: dict,  # mutable-ok: signature dictated by BaseEmbeddingConfig
    ) -> dict:  # mutable-ok: signature dictated by BaseEmbeddingConfig
        if isinstance(input, str):
            input = [input]  # rebind-ok: normalize locally, mirrors hosted_vllm  # mutable-ok: API request payload

        return {  # mutable-ok: API request payload
            "model": model,
            "input": input,
            **optional_params,
        }

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        request_data: dict,  # mutable-ok: signature dictated by BaseEmbeddingConfig
        optional_params: dict,  # mutable-ok: signature dictated by BaseEmbeddingConfig
        litellm_params: dict,  # mutable-ok: signature dictated by BaseEmbeddingConfig
    ) -> EmbeddingResponse:
        logging_obj.post_call(original_response=raw_response.text)

        response_json: Final = raw_response.json()

        return convert_to_model_response_object(
            response_object=response_json,
            model_response_object=model_response,
            response_type="embedding",
        )

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: signature dictated by BaseEmbeddingConfig
        return [  # mutable-ok: supported-params list, mirrors hosted_vllm
            "timeout",
            "dimensions",
            "encoding_format",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: signature dictated by BaseEmbeddingConfig
        optional_params: dict,  # mutable-ok: signature dictated by BaseEmbeddingConfig
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: signature dictated by BaseEmbeddingConfig
        mapped: Final = {  # mutable-ok: API request payload
            k: v for k, v in non_default_params.items() if k in self.get_supported_openai_params(model)
        }
        return {**optional_params, **mapped}  # mutable-ok: API request payload

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: signature dictated by BaseEmbeddingConfig
    ) -> BaseLLMException:
        return DockerModelRunnerEmbeddingError(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
