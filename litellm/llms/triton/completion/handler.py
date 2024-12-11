import json
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import httpx  # type: ignore

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    custom_prompt,
    prompt_factory,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.utils import (
    Choices,
    CustomStreamWrapper,
    Delta,
    EmbeddingResponse,
    Message,
    ModelResponse,
    Usage,
    map_finish_reason,
)

from ...base import BaseLLM
from ..common_utils import TritonError


class TritonChatCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    async def aembedding(
        self,
        data: dict,
        model_response: litellm.utils.EmbeddingResponse,
        api_base: str,
        logging_obj: Any,
        api_key: Optional[str] = None,
    ) -> EmbeddingResponse:
        async_handler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.TRITON, params={"timeout": 600.0}
        )

        response = await async_handler.post(url=api_base, data=json.dumps(data))

        if response.status_code != 200:
            raise TritonError(status_code=response.status_code, message=response.text)

        _text_response = response.text

        logging_obj.post_call(original_response=_text_response)

        _json_response = response.json()
        _embedding_output = []

        _outputs = _json_response["outputs"]
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

        model_response.model = _json_response.get("model_name", "None")
        model_response.data = _embedding_output

        return model_response

    async def embedding(
        self,
        model: str,
        input: List[str],
        timeout: float,
        api_base: str,
        model_response: litellm.utils.EmbeddingResponse,
        logging_obj: Any,
        optional_params: dict,
        api_key: Optional[str] = None,
        client=None,
        aembedding: bool = False,
    ) -> EmbeddingResponse:
        data_for_triton = {
            "inputs": [
                {
                    "name": "input_text",
                    "shape": [len(input)],
                    "datatype": "BYTES",
                    "data": input,
                }
            ]
        }

        curl_string = f"curl {api_base} -X POST -H 'Content-Type: application/json' -d '{data_for_triton}'"

        logging_obj.pre_call(
            input="",
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": curl_string,
            },
        )

        if aembedding:
            response = await self.aembedding(  # type: ignore
                data=data_for_triton,
                model_response=model_response,
                logging_obj=logging_obj,
                api_base=api_base,
                api_key=api_key,
            )
            return response
        else:
            raise Exception(
                "Only async embedding supported for triton, please use litellm.aembedding() for now"
            )

    def completion(
        self,
        model: str,
        messages: List,
        timeout: float,
        api_base: str,
        logging_obj: Any,
        optional_params: dict,
        litellm_params: dict,
        model_response: ModelResponse,
        api_key: Optional[str] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
        stream: Optional[bool] = False,
        acompletion: bool = False,
        headers: Optional[dict] = None,
    ) -> ModelResponse:
        type_of_model = ""
        optional_params.pop("stream", False)
        if api_base.endswith("generate"):  ### This is a trtllm model
            data_for_triton = litellm.TritonConfig().transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers or {},
            )
            type_of_model = "trtllm"

        elif api_base.endswith(
            "infer"
        ):  ### This is an infer model with a custom model on triton
            text_input = messages[0]["content"]
            data_for_triton = {
                "inputs": [
                    {
                        "name": "text_input",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [text_input],
                    }
                ]
            }

            for k, v in optional_params.items():
                if not (k == "stream" or k == "max_retries"):
                    datatype = "INT32" if isinstance(v, int) else "BYTES"
                    datatype = "FP32" if isinstance(v, float) else datatype
                    data_for_triton["inputs"].append(
                        {"name": k, "shape": [1], "datatype": datatype, "data": [v]}
                    )

            if "max_tokens" not in optional_params:
                data_for_triton["inputs"].append(
                    {
                        "name": "max_tokens",
                        "shape": [1],
                        "datatype": "INT32",
                        "data": [20],
                    }
                )

            type_of_model = "infer"
        else:  ## Unknown model type passthrough
            data_for_triton = {
                "inputs": [
                    {
                        "name": "text_input",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [messages[0]["content"]],
                    }
                ]
            }

        if logging_obj:
            logging_obj.pre_call(
                input=messages,
                api_key=api_key,
                additional_args={
                    "complete_input_dict": optional_params,
                    "api_base": api_base,
                    "http_client": client,
                },
            )

        headers = litellm.TritonConfig().validate_environment(
            headers=headers or {},
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
        )
        json_data_for_triton: str = json.dumps(data_for_triton)

        if acompletion:
            return self.acompletion(  # type: ignore
                model,
                json_data_for_triton,
                headers=headers,
                logging_obj=logging_obj,
                api_base=api_base,
                stream=stream,
                model_response=model_response,
                type_of_model=type_of_model,
            )

        if client is None or not isinstance(client, HTTPHandler):
            handler = _get_httpx_client()
        else:
            handler = client

        if stream:
            return self._handle_stream(  # type: ignore
                handler, api_base, json_data_for_triton, model, logging_obj
            )
        else:
            response = handler.post(
                url=api_base, data=json_data_for_triton, headers=headers
            )
            return self._handle_response(
                response, model_response, logging_obj, type_of_model=type_of_model
            )

    async def acompletion(
        self,
        model: str,
        data_for_triton,
        api_base,
        stream,
        logging_obj,
        headers,
        model_response,
        type_of_model,
    ) -> ModelResponse:
        handler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.TRITON, params={"timeout": 600.0}
        )
        if stream:
            return self._ahandle_stream(  # type: ignore
                handler, api_base, data_for_triton, model, logging_obj
            )
        else:
            response = await handler.post(
                url=api_base, data=data_for_triton, headers=headers
            )

            return self._handle_response(
                response, model_response, logging_obj, type_of_model=type_of_model
            )

    def _handle_stream(self, handler, api_base, data_for_triton, model, logging_obj):
        response = handler.post(
            url=api_base + "_stream", data=data_for_triton, stream=True
        )
        streamwrapper = litellm.CustomStreamWrapper(
            response.iter_lines(),
            model=model,
            custom_llm_provider="triton",
            logging_obj=logging_obj,
        )
        for chunk in streamwrapper:
            yield (chunk)

    async def _ahandle_stream(
        self, handler, api_base, data_for_triton, model, logging_obj
    ):
        response = await handler.post(
            url=api_base + "_stream", data=data_for_triton, stream=True
        )
        streamwrapper = litellm.CustomStreamWrapper(
            response.aiter_lines(),
            model=model,
            custom_llm_provider="triton",
            logging_obj=logging_obj,
        )
        async for chunk in streamwrapper:
            yield (chunk)

    def _handle_response(self, response, model_response, logging_obj, type_of_model):
        if logging_obj:
            logging_obj.post_call(original_response=response)

        if response.status_code != 200:
            raise TritonError(status_code=response.status_code, message=response.text)

        _json_response = response.json()
        model_response.model = _json_response.get("model_name", "None")
        if type_of_model == "trtllm":
            model_response.choices = [
                Choices(index=0, message=Message(content=_json_response["text_output"]))
            ]
        elif type_of_model == "infer":
            model_response.choices = [
                Choices(
                    index=0,
                    message=Message(content=_json_response["outputs"][0]["data"]),
                )
            ]
        else:
            model_response.choices = [
                Choices(index=0, message=Message(content=_json_response["outputs"]))
            ]
        return model_response

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
