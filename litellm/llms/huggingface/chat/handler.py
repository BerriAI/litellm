## Uses the huggingface text generation inference API
import json
import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    get_args,
)

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.huggingface.chat.transformation import (
    HuggingfaceChatConfig as HuggingfaceConfig,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import EmbeddingResponse
from litellm.types.utils import Logprobs as TextCompletionLogprobs
from litellm.types.utils import ModelResponse

from ...base import BaseLLM
from ..common_utils import HuggingfaceError

hf_chat_config = HuggingfaceConfig()


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

    model_info = http_client.get(url=api_base)

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

    model_info = await http_client.get(url=api_base)

    model_info_dict = model_info.json()

    pipeline_tag: Optional[str] = model_info_dict.get("pipeline_tag", None)

    return pipeline_tag


async def make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    timeout: Optional[Union[float, httpx.Timeout]],
    json_mode: bool,
) -> Tuple[Any, httpx.Headers]:
    if client is None:
        client = litellm.module_level_aclient

    try:
        response = await client.post(
            api_base, headers=headers, data=data, stream=True, timeout=timeout
        )
    except httpx.HTTPStatusError as e:
        error_headers = getattr(e, "headers", None)
        error_response = getattr(e, "response", None)
        if error_headers is None and error_response:
            error_headers = getattr(error_response, "headers", None)
        raise HuggingfaceError(
            status_code=e.response.status_code,
            message=str(await e.response.aread()),
            headers=cast(dict, error_headers) if error_headers else None,
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise HuggingfaceError(status_code=500, message=str(e))

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=response,  # Pass the completion stream for logging
        additional_args={"complete_input_dict": data},
    )

    return response.aiter_lines(), response.headers


class Huggingface(BaseLLM):
    _client_session: Optional[httpx.Client] = None
    _aclient_session: Optional[httpx.AsyncClient] = None

    def __init__(self) -> None:
        super().__init__()

    def completion(  # noqa: PLR0915
        self,
        model: str,
        messages: list,
        api_base: Optional[str],
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: float,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        litellm_params: dict,
        custom_prompt_dict={},
        acompletion: bool = False,
        logger_fn=None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        headers: dict = {},
    ):
        super().completion()
        exception_mapping_worked = False
        try:
            task, model = hf_chat_config.get_hf_task_for_model(model)
            litellm_params["task"] = task
            headers = hf_chat_config.validate_environment(
                api_key=api_key,
                headers=headers,
                model=model,
                messages=messages,
                optional_params=optional_params,
            )
            completion_url = hf_chat_config.get_api_base(api_base=api_base, model=model)
            data = hf_chat_config.transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )

            ## LOGGING
            logging_obj.pre_call(
                input=data,
                api_key=api_key,
                additional_args={
                    "complete_input_dict": data,
                    "headers": headers,
                    "api_base": completion_url,
                    "acompletion": acompletion,
                },
            )
            ## COMPLETION CALL

            if acompletion is True:
                ### ASYNC STREAMING
                if optional_params.get("stream", False):
                    return self.async_streaming(logging_obj=logging_obj, api_base=completion_url, data=data, headers=headers, model_response=model_response, model=model, timeout=timeout, messages=messages)  # type: ignore
                else:
                    ### ASYNC COMPLETION
                    return self.acompletion(
                        api_base=completion_url,
                        data=data,
                        headers=headers,
                        model_response=model_response,
                        encoding=encoding,
                        model=model,
                        optional_params=optional_params,
                        timeout=timeout,
                        litellm_params=litellm_params,
                        logging_obj=logging_obj,
                        api_key=api_key,
                        messages=messages,
                        client=(
                            client
                            if client is not None
                            and isinstance(client, AsyncHTTPHandler)
                            else None
                        ),
                    )
            if client is None or not isinstance(client, HTTPHandler):
                client = _get_httpx_client()
            ### SYNC STREAMING
            if "stream" in optional_params and optional_params["stream"] is True:
                response = client.post(
                    url=completion_url,
                    headers=headers,
                    data=json.dumps(data),
                    stream=optional_params["stream"],
                )
                return response.iter_lines()
            ### SYNC COMPLETION
            else:
                response = client.post(
                    url=completion_url,
                    headers=headers,
                    data=json.dumps(data),
                )

                return hf_chat_config.transform_response(
                    model=model,
                    raw_response=response,
                    model_response=model_response,
                    logging_obj=logging_obj,
                    api_key=api_key,
                    request_data=data,
                    messages=messages,
                    optional_params=optional_params,
                    encoding=encoding,
                    json_mode=None,
                    litellm_params=litellm_params,
                )
        except httpx.HTTPStatusError as e:
            raise HuggingfaceError(
                status_code=e.response.status_code,
                message=e.response.text,
                headers=e.response.headers,
            )
        except HuggingfaceError as e:
            exception_mapping_worked = True
            raise e
        except Exception as e:
            if exception_mapping_worked:
                raise e
            else:
                import traceback

                raise HuggingfaceError(status_code=500, message=traceback.format_exc())

    async def acompletion(
        self,
        api_base: str,
        data: dict,
        headers: dict,
        model_response: ModelResponse,
        encoding: Any,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        timeout: float,
        logging_obj: LiteLLMLoggingObj,
        api_key: str,
        messages: List[AllMessageValues],
        client: Optional[AsyncHTTPHandler] = None,
    ):
        response: Optional[httpx.Response] = None
        try:
            if client is None:
                client = get_async_httpx_client(
                    llm_provider=litellm.LlmProviders.HUGGINGFACE
                )
            ### ASYNC COMPLETION
            http_response = await client.post(
                url=api_base, headers=headers, data=json.dumps(data), timeout=timeout
            )

            response = http_response

            return hf_chat_config.transform_response(
                model=model,
                raw_response=http_response,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key=api_key,
                request_data=data,
                messages=messages,
                optional_params=optional_params,
                encoding=encoding,
                json_mode=None,
                litellm_params=litellm_params,
            )
        except Exception as e:
            if isinstance(e, httpx.TimeoutException):
                raise HuggingfaceError(status_code=500, message="Request Timeout Error")
            elif isinstance(e, HuggingfaceError):
                raise e
            elif response is not None and hasattr(response, "text"):
                raise HuggingfaceError(
                    status_code=500,
                    message=f"{str(e)}\n\nOriginal Response: {response.text}",
                    headers=response.headers,
                )
            else:
                raise HuggingfaceError(status_code=500, message=f"{str(e)}")

    async def async_streaming(
        self,
        logging_obj,
        api_base: str,
        data: dict,
        headers: dict,
        model_response: ModelResponse,
        messages: List[AllMessageValues],
        model: str,
        timeout: float,
        client: Optional[AsyncHTTPHandler] = None,
    ):
        completion_stream, _ = await make_call(
            client=client,
            api_base=api_base,
            headers=headers,
            data=json.dumps(data),
            model=model,
            messages=messages,
            logging_obj=logging_obj,
            timeout=timeout,
            json_mode=False,
        )
        streamwrapper = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="huggingface",
            logging_obj=logging_obj,
        )
        return streamwrapper

    def _transform_input_on_pipeline_tag(
        self, input: List, pipeline_tag: Optional[str]
    ) -> dict:
        if pipeline_tag is None:
            return {"inputs": input}
        if pipeline_tag == "sentence-similarity" or pipeline_tag == "similarity":
            if len(input) < 2:
                raise HuggingfaceError(
                    status_code=400,
                    message="sentence-similarity requires 2+ sentences",
                )
            return {"inputs": {"source_sentence": input[0], "sentences": input[1:]}}
        elif pipeline_tag == "rerank":
            if len(input) < 2:
                raise HuggingfaceError(
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
            model=model, task_type=task_type, api_base=embed_url
        )

        data = self._transform_input_on_pipeline_tag(input=input, pipeline_tag=hf_task)

        if len(optional_params.keys()) > 0:
            data["options"] = optional_params

        return data

    def _process_optional_params(self, data: dict, optional_params: dict) -> dict:
        special_options_keys = HuggingfaceConfig().get_special_options_params()
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
                raise HuggingfaceError(
                    status_code=400,
                    message="sentence transformers requires 2+ sentences",
                )
            data = {"inputs": {"source_sentence": input[0], "sentences": input[1:]}}
        else:
            data = {"inputs": input}

            task_type = optional_params.pop("input_type", None)

            if call_type == "sync":
                hf_task = get_hf_task_embedding_for_model(
                    model=model, task_type=task_type, api_base=embed_url
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
            raise HuggingfaceError(status_code=500, message=embeddings["error"])

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
        headers = hf_chat_config.validate_environment(
            api_key=api_key,
            headers=headers,
            model=model,
            optional_params=optional_params,
            messages=[],
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
            embed_url = f"https://api-inference.huggingface.co/models/{model}"

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
            raise HuggingfaceError(status_code=500, message=embeddings["error"])

        ## PROCESS RESPONSE ##
        return self._process_embedding_response(
            embeddings=embeddings,
            model_response=model_response,
            model=model,
            input=input,
            encoding=encoding,
        )

    def _transform_logprobs(
        self, hf_response: Optional[List]
    ) -> Optional[TextCompletionLogprobs]:
        """
        Transform Hugging Face logprobs to OpenAI.Completion() format
        """
        if hf_response is None:
            return None

        # Initialize an empty list for the transformed logprobs
        _logprob: TextCompletionLogprobs = TextCompletionLogprobs(
            text_offset=[],
            token_logprobs=[],
            tokens=[],
            top_logprobs=[],
        )

        # For each Hugging Face response, transform the logprobs
        for response in hf_response:
            # Extract the relevant information from the response
            response_details = response["details"]
            top_tokens = response_details.get("top_tokens", {})

            for i, token in enumerate(response_details["prefill"]):
                # Extract the text of the token
                token_text = token["text"]

                # Extract the logprob of the token
                token_logprob = token["logprob"]

                # Add the token information to the 'token_info' list
                cast(List[str], _logprob.tokens).append(token_text)
                cast(List[float], _logprob.token_logprobs).append(token_logprob)

                # stub this to work with llm eval harness
                top_alt_tokens = {"": -1.0, "": -2.0, "": -3.0}  # noqa: F601
                cast(List[Dict[str, float]], _logprob.top_logprobs).append(
                    top_alt_tokens
                )

            # For each element in the 'tokens' list, extract the relevant information
            for i, token in enumerate(response_details["tokens"]):
                # Extract the text of the token
                token_text = token["text"]

                # Extract the logprob of the token
                token_logprob = token["logprob"]

                top_alt_tokens = {}
                temp_top_logprobs = []
                if top_tokens != {}:
                    temp_top_logprobs = top_tokens[i]

                # top_alt_tokens should look like this: { "alternative_1": -1, "alternative_2": -2, "alternative_3": -3 }
                for elem in temp_top_logprobs:
                    text = elem["text"]
                    logprob = elem["logprob"]
                    top_alt_tokens[text] = logprob

                # Add the token information to the 'token_info' list
                cast(List[str], _logprob.tokens).append(token_text)
                cast(List[float], _logprob.token_logprobs).append(token_logprob)
                cast(List[Dict[str, float]], _logprob.top_logprobs).append(
                    top_alt_tokens
                )

                # Add the text offset of the token
                # This is computed as the sum of the lengths of all previous tokens
                cast(List[int], _logprob.text_offset).append(
                    sum(len(t["text"]) for t in response_details["tokens"][:i])
                )

        return _logprob
