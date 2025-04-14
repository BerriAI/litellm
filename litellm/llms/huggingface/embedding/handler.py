import json
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Union, get_args

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.utils import EmbeddingResponse

from ...base import BaseLLM
from ..common_utils import HuggingFaceError
from .transformation import HuggingFaceEmbeddingConfig

config = HuggingFaceEmbeddingConfig()

HF_HUB_URL = "https://huggingface.co"

hf_tasks_embeddings = Literal[  # pipeline tags + hf tei endpoints - https://huggingface.github.io/text-embeddings-inference/#/
    "sentence-similarity", "feature-extraction", "rerank", "embed", "similarity"
]


def get_hf_task_embedding_for_model(
    model: str, task_type: Optional[str], api_base: str
) -> Optional[str]:
    if task_type is not None:
        if task_type in get_args(hf_tasks_embeddings):
            return task_type
        else:
            raise Exception(
                "Invalid task_type={}. Expected one of={}".format(
                    task_type, hf_tasks_embeddings
                )
            )
    http_client = HTTPHandler(concurrent_limit=1)

    model_info = http_client.get(url=f"{api_base}/api/models/{model}")

    model_info_dict = model_info.json()

    pipeline_tag: Optional[str] = model_info_dict.get("pipeline_tag", None)

    return pipeline_tag


async def async_get_hf_task_embedding_for_model(
    model: str, task_type: Optional[str], api_base: str
) -> Optional[str]:
    if task_type is not None:
        if task_type in get_args(hf_tasks_embeddings):
            return task_type
        else:
            raise Exception(
                "Invalid task_type={}. Expected one of={}".format(
                    task_type, hf_tasks_embeddings
                )
            )
    http_client = get_async_httpx_client(
        llm_provider=litellm.LlmProviders.HUGGINGFACE,
    )

    model_info = await http_client.get(url=f"{api_base}/api/models/{model}")

    model_info_dict = model_info.json()

    pipeline_tag: Optional[str] = model_info_dict.get("pipeline_tag", None)

    return pipeline_tag


class HuggingFaceEmbedding(BaseLLM):
    _client_session: Optional[httpx.Client] = None
    _aclient_session: Optional[httpx.AsyncClient] = None

    def __init__(self) -> None:
        super().__init__()

    def _transform_input_on_pipeline_tag(
        self, input: List, pipeline_tag: Optional[str]
    ) -> dict:
        if pipeline_tag is None:
            return {"inputs": input}
        if pipeline_tag == "sentence-similarity" or pipeline_tag == "similarity":
            if len(input) < 2:
                raise HuggingFaceError(
                    status_code=400,
                    message="sentence-similarity requires 2+ sentences",
                )
            return {"inputs": {"source_sentence": input[0], "sentences": input[1:]}}
        elif pipeline_tag == "rerank":
            if len(input) < 2:
                raise HuggingFaceError(
                    status_code=400,
                    message="reranker requires 2+ sentences",
                )
            return {"inputs": {"query": input[0], "texts": input[1:]}}
        return {"inputs": input}  # default to feature-extraction pipeline tag

    async def _async_transform_input(
        self,
        model: str,
        task_type: Optional[str],
        embed_url: str,
        input: List,
        optional_params: dict,
    ) -> dict:
        hf_task = await async_get_hf_task_embedding_for_model(
            model=model, task_type=task_type, api_base=HF_HUB_URL
        )

        data = self._transform_input_on_pipeline_tag(input=input, pipeline_tag=hf_task)

        if len(optional_params.keys()) > 0:
            data["options"] = optional_params

        return data

    def _process_optional_params(self, data: dict, optional_params: dict) -> dict:
        special_options_keys = config.get_special_options_params()
        special_parameters_keys = [
            "min_length",
            "max_length",
            "top_k",
            "top_p",
            "temperature",
            "repetition_penalty",
            "max_time",
        ]

        for k, v in optional_params.items():
            if k in special_options_keys:
                data.setdefault("options", {})
                data["options"][k] = v
            elif k in special_parameters_keys:
                data.setdefault("parameters", {})
                data["parameters"][k] = v
            else:
                data[k] = v

        return data

    def _transform_input(
        self,
        input: List,
        model: str,
        call_type: Literal["sync", "async"],
        optional_params: dict,
        embed_url: str,
    ) -> dict:
        data: Dict = {}

        ## TRANSFORMATION ##
        if "sentence-transformers" in model:
            if len(input) == 0:
                raise HuggingFaceError(
                    status_code=400,
                    message="sentence transformers requires 2+ sentences",
                )
            data = {"inputs": {"source_sentence": input[0], "sentences": input[1:]}}
        else:
            data = {"inputs": input}

            task_type = optional_params.pop("input_type", None)

            if call_type == "sync":
                hf_task = get_hf_task_embedding_for_model(
                    model=model, task_type=task_type, api_base=HF_HUB_URL
                )
            elif call_type == "async":
                return self._async_transform_input(
                    model=model, task_type=task_type, embed_url=embed_url, input=input
                )  # type: ignore

            data = self._transform_input_on_pipeline_tag(
                input=input, pipeline_tag=hf_task
            )

        if len(optional_params.keys()) > 0:
            data = self._process_optional_params(
                data=data, optional_params=optional_params
            )

        return data

    def _process_embedding_response(
        self,
        embeddings: dict,
        model_response: EmbeddingResponse,
        model: str,
        input: List,
        encoding: Any,
    ) -> EmbeddingResponse:
        output_data = []
        if "similarities" in embeddings:
            for idx, embedding in embeddings["similarities"]:
                output_data.append(
                    {
                        "object": "embedding",
                        "index": idx,
                        "embedding": embedding,  # flatten list returned from hf
                    }
                )
        else:
            for idx, embedding in enumerate(embeddings):
                if isinstance(embedding, float):
                    output_data.append(
                        {
                            "object": "embedding",
                            "index": idx,
                            "embedding": embedding,  # flatten list returned from hf
                        }
                    )
                elif isinstance(embedding, list) and isinstance(embedding[0], float):
                    output_data.append(
                        {
                            "object": "embedding",
                            "index": idx,
                            "embedding": embedding,  # flatten list returned from hf
                        }
                    )
                else:
                    output_data.append(
                        {
                            "object": "embedding",
                            "index": idx,
                            "embedding": embedding[0][
                                0
                            ],  # flatten list returned from hf
                        }
                    )
        model_response.object = "list"
        model_response.data = output_data
        model_response.model = model
        input_tokens = 0
        for text in input:
            input_tokens += len(encoding.encode(text))

        setattr(
            model_response,
            "usage",
            litellm.Usage(
                prompt_tokens=input_tokens,
                completion_tokens=input_tokens,
                total_tokens=input_tokens,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
        )
        return model_response

    async def aembedding(
        self,
        model: str,
        input: list,
        model_response: litellm.utils.EmbeddingResponse,
        timeout: Union[float, httpx.Timeout],
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        api_base: str,
        api_key: Optional[str],
        headers: dict,
        encoding: Callable,
        client: Optional[AsyncHTTPHandler] = None,
    ):
        ## TRANSFORMATION ##
        data = self._transform_input(
            input=input,
            model=model,
            call_type="sync",
            optional_params=optional_params,
            embed_url=api_base,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "headers": headers,
                "api_base": api_base,
            },
        )
        ## COMPLETION CALL
        if client is None:
            client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.HUGGINGFACE,
            )

        response = await client.post(api_base, headers=headers, data=json.dumps(data))

        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=response,
        )

        embeddings = response.json()

        if "error" in embeddings:
            raise HuggingFaceError(status_code=500, message=embeddings["error"])

        ## PROCESS RESPONSE ##
        return self._process_embedding_response(
            embeddings=embeddings,
            model_response=model_response,
            model=model,
            input=input,
            encoding=encoding,
        )

    def embedding(
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        optional_params: dict,
        litellm_params: dict,
        logging_obj: LiteLLMLoggingObj,
        encoding: Callable,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Union[float, httpx.Timeout] = httpx.Timeout(None),
        aembedding: Optional[bool] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        headers={},
    ) -> EmbeddingResponse:
        super().embedding()
        headers = config.validate_environment(
            api_key=api_key,
            headers=headers,
            model=model,
            optional_params=optional_params,
            messages=[],
            litellm_params=litellm_params,
        )
        task_type = optional_params.pop("input_type", None)
        task = get_hf_task_embedding_for_model(
            model=model, task_type=task_type, api_base=HF_HUB_URL
        )
        # print_verbose(f"{model}, {task}")
        embed_url = ""
        if "https" in model:
            embed_url = model
        elif api_base:
            embed_url = api_base
        elif "HF_API_BASE" in os.environ:
            embed_url = os.getenv("HF_API_BASE", "")
        elif "HUGGINGFACE_API_BASE" in os.environ:
            embed_url = os.getenv("HUGGINGFACE_API_BASE", "")
        else:
            embed_url = (
                f"https://router.huggingface.co/hf-inference/pipeline/{task}/{model}"
            )

        ## ROUTING ##
        if aembedding is True:
            return self.aembedding(
                input=input,
                model_response=model_response,
                timeout=timeout,
                logging_obj=logging_obj,
                headers=headers,
                api_base=embed_url,  # type: ignore
                api_key=api_key,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
                model=model,
                optional_params=optional_params,
                encoding=encoding,
            )

        ## TRANSFORMATION ##

        data = self._transform_input(
            input=input,
            model=model,
            call_type="sync",
            optional_params=optional_params,
            embed_url=embed_url,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "headers": headers,
                "api_base": embed_url,
            },
        )
        ## COMPLETION CALL
        if client is None or not isinstance(client, HTTPHandler):
            client = HTTPHandler(concurrent_limit=1)
        response = client.post(embed_url, headers=headers, data=json.dumps(data))

        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=response,
        )

        embeddings = response.json()

        if "error" in embeddings:
            raise HuggingFaceError(status_code=500, message=embeddings["error"])

        ## PROCESS RESPONSE ##
        return self._process_embedding_response(
            embeddings=embeddings,
            model_response=model_response,
            model=model,
            input=input,
            encoding=encoding,
        )
