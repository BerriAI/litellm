import os
import json
from enum import Enum
import requests
import time
from typing import Callable, Optional, List, Sequence, Any, Union, Dict
from litellm.utils import ModelResponse, Choices, Usage, map_finish_reason, CustomStreamWrapper, Message, EmbeddingResponse
import litellm
from .prompt_templates.factory import prompt_factory, custom_prompt
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from .base import BaseLLM
import httpx
from typing import Union,Collection


class TritonError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
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
    ) -> EmbeddingResponse:
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
        _output_data = [output["data"] for output in _outputs]
        _embedding_output = {
            "object": "embedding",
            "index": 0,
            "embedding": _output_data,
        }

        model_response.model = _json_response.get("model_name", "None")
        model_response.data = [_embedding_output]

        return model_response

    async def embedding(
        self,
        model: str,
        input: List[str],
        timeout: float,
        api_base: str,
        model_response: litellm.utils.EmbeddingResponse,
        api_key: Optional[str] = None,
        logging_obj=None,
        optional_params=None,
        client=None,
        aembedding: bool = False,
    ) -> EmbeddingResponse:
        data_for_triton = {
            "inputs": [
                {
                    "name": "input_text",
                    "shape": [len(input)],  # size of the input data
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
            response = await self.aembedding(
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
        messages: List[dict],
        timeout: float,
        api_base: str,
        model_response: ModelResponse,
        api_key: Optional[str] = None,
        logging_obj=None,
        optional_params=None,
        client=None,
        stream: bool = False,
    ) -> ModelResponse:

        type_of_model = ""
        if api_base.endswith("generate"):  ### This is a trtllm model
            text_input = messages[0]["content"]
            data_for_triton: Dict[str, Any]  = {
                "text_input": str(text_input),
                "parameters": {
                    "max_tokens": int(optional_params.get("max_tokens", 20)),
                    "bad_words": [""],
                    "stop_words": [""]
                }
            }
            data_for_triton["parameters"].update( optional_params)
            type_of_model = "trtllm"

        elif api_base.endswith("infer"):  ### This is an infer model with a custom model on triton
            text_input = messages[0]["content"]
            data_for_triton = {
                "inputs": [{"name": "text_input", "shape": [1], "datatype": "BYTES", "data": [text_input]}]
            }

            for k, v in optional_params.items():
                if not (k == "stream" or k == "max_retries"):
                    datatype = "INT32" if isinstance(v, int) else "BYTES"
                    datatype = "FP32" if isinstance(v, float) else datatype
                    data_for_triton['inputs'].append({"name": k, "shape": [1], "datatype": datatype, "data": [v]})

            if "max_tokens" not in optional_params:
                data_for_triton['inputs'].append({"name": "max_tokens", "shape": [1], "datatype": "INT32", "data": [20]})

            type_of_model = "infer"
        else:  ## Unknown model type passthrough
            data_for_triton = {
                "inputs": [{"name": "text_input", "shape": [1], "datatype": "BYTES", "data": [messages[0]["content"]]}] 
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
        handler = requests.Session()
        handler.timeout = (600.0, 5.0)

        response = handler.post(url=api_base, json=data_for_triton)

        if logging_obj:
            logging_obj.post_call(original_response=response)

        if response.status_code != 200:
            raise TritonError(status_code=response.status_code, message=response.text)
        _json_response = response.json()

        model_response.model = _json_response.get("model_name", "None")
        if type_of_model == "trtllm":
            model_response.choices = [Choices(index=0, message=Message(content=_json_response['text_output']))]
        elif type_of_model == "infer":
            model_response.choices = [Choices(index=0, message=Message(content=_json_response['outputs'][0]['data']))]
        else:
            model_response.choices = [Choices(index=0, message=Message(content=_json_response['outputs']))]

        return model_response
