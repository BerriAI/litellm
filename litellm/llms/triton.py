import os, types
import json
from enum import Enum
import requests, copy  # type: ignore
import time
from typing import Callable, Optional, List
import litellm
from .prompt_templates.factory import prompt_factory, custom_prompt
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from .base import BaseLLM
import httpx  # type: ignore


class TritonError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST",
            url="https://api.anthropic.com/v1/messages",  # using anthropic api base since httpx requires a url
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class TritonChatCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    async def aembedding(
        self,
        data: dict,
        model_response: litellm.utils.EmbeddingResponse,
        api_base: str,
        logging_obj=None,
        api_key: Optional[str] = None,
    ):

        async_handler = AsyncHTTPHandler(
            timeout=httpx.Timeout(timeout=600.0, connect=5.0)
        )

        response = await async_handler.post(url=api_base, data=json.dumps(data))

        if response.status_code != 200:
            raise TritonError(status_code=response.status_code, message=response.text)

        _text_response = response.text

        logging_obj.post_call(original_response=_text_response)

        _json_response = response.json()

        _outputs = _json_response["outputs"]
        _output_data = _outputs[0]["data"]
        _embedding_output = {
            "object": "embedding",
            "index": 0,
            "embedding": _output_data,
        }

        model_response.model = _json_response.get("model_name", "None")
        model_response.data = [_embedding_output]

        return model_response

    def embedding(
        self,
        model: str,
        input: list,
        timeout: float,
        api_base: str,
        model_response: litellm.utils.EmbeddingResponse,
        api_key: Optional[str] = None,
        logging_obj=None,
        optional_params=None,
        client=None,
        aembedding=None,
    ):
        data_for_triton = {
            "inputs": [
                {
                    "name": "input_text",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": input,
                }
            ]
        }

        ## LOGGING

        curl_string = f"curl {api_base} -X POST -H 'Content-Type: application/json' -d '{data_for_triton}'"

        logging_obj.pre_call(
            input="",
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": curl_string,
            },
        )

        if aembedding == True:
            response = self.aembedding(
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
