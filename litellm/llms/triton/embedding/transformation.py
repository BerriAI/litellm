from typing import List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import AllMessageValues, BaseLLMException
from litellm.llms.base_llm.embedding.transformation import (
    BaseEmbeddingConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse, Usage
from litellm.utils import token_counter

from ..common_utils import TritonError


class TritonEmbeddingConfig(BaseEmbeddingConfig):
    """
    Transformations for triton /embeddings endpoint (This is a trtllm model)
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self, model: str) -> list:
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to Triton Embedding params
        """
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return {}

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        return {
            "inputs": [
                {
                    "name": "input_text",
                    "shape": [len(input)],
                    "datatype": "BYTES",
                    "data": input,
                }
            ]
        }

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> EmbeddingResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise TritonError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        _embedding_output = []

        _outputs = raw_response_json["outputs"]
        for output in _outputs:
            _shape = output["shape"]
            _data = output["data"]
            _split_output_data = self.split_embedding_by_shape(_data, _shape)

            for idx, embedding in enumerate(_split_output_data):
                _embedding_output.append(
                    {
                        "object": "embedding",
                        "index": idx,
                        "embedding": embedding,
                    }
                )

        model_response.model = raw_response_json.get("model_name", "None")
        model_response.data = _embedding_output
        model_response.usage = self._build_embedding_usage(
            model=model, request_data=request_data
        )
        return model_response

    def _build_embedding_usage(self, model: str, request_data: dict) -> Usage:
        input_data = request_data.get("inputs", [])
        input_text_values: List[str] = []
        for item in input_data:
            if isinstance(item, dict) and item.get("name") == "input_text":
                data_values = item.get("data", [])
                if isinstance(data_values, list):
                    input_text_values = [str(value) for value in data_values]
                break

        prompt_tokens = 0
        for text in input_text_values:
            if not text:
                continue
            try:
                prompt_tokens += token_counter(model=model, text=text)
            except Exception:
                prompt_tokens += len(text.split())

        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return TritonError(
            message=error_message, status_code=status_code, headers=headers
        )

    @staticmethod
    def split_embedding_by_shape(
        data: List[float], shape: List[int]
    ) -> List[List[float]]:
        if len(shape) != 2:
            raise ValueError("Shape must be of length 2.")
        embedding_size = shape[1]
        return [
            data[i * embedding_size : (i + 1) * embedding_size] for i in range(shape[0])
        ]
