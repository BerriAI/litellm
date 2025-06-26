# What is this?
## Controller file for Predibase Integration - https://predibase.com/
from typing import Any, Dict
from dataclasses import dataclass, field
import json
import time
from functools import partial
from typing import Callable, Optional, Union

import httpx  # type: ignore

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)
from litellm.types.utils import LiteLLMLoggingBaseClass
from litellm.utils import CustomStreamWrapper, ModelResponse

from ..common_utils import BytezError

CUSTOM_LLM_PROVIDER = "bytez"

API_BASE = "https://api.bytez.com/models/v2"


@dataclass
class CacheEntry:
    value: Any
    lru_counter: int


@dataclass
class SupportedModelCache:
    max_items: int = 100
    cache: Dict[str, CacheEntry] = field(default_factory=dict, init=True)
    counter: int = 0

    @property
    def item_count(self):
        return len(self.cache.keys())

    def increment_counter(self):
        self.counter += 1
        return self.counter

    def get(self, key: str) -> Any:
        cache_entry = self.cache[key]

        cache_entry.lru_counter = self.increment_counter()

        return cache_entry.value

    def has(self, key: str) -> bool:
        cache_entry = self.cache.get(key)

        return bool(cache_entry)

    def set(self, key: str, value: Any) -> None:
        lru_counter = self.increment_counter()

        cache_entry = CacheEntry(value=value, lru_counter=lru_counter)

        # evict
        if self.item_count + 1 > self.max_items:
            # find the oldest lru stamp and remove it
            key, cache_entry = min(
                [(key, cache_entry) for key, cache_entry in self.cache.items()],
                key=lambda item: item[1].lru_counter,
            )

            del self.cache[key]

        self.cache[key] = cache_entry


class BytezChatCompletion:
    def __init__(self) -> None:
        super().__init__()
        self.cache = SupportedModelCache()

    def to_model_id(self, model):
        # remove bytez off of the front of the model
        model = "/".join(model.split("/")[1:])
        return model

    def validate_model_is_supported(self, model_id: str, headers: Dict):
        if self.cache.has(model_id):
            return self.cache.get(model_id)

        url = f"{API_BASE}/list/models?modelId={model_id}"

        response = httpx.request(method="GET", url=url, headers=headers)

        json = response.json()

        error = json.get("error")

        if error:
            raise Exception(error)

        models = json.get("output")

        if len(models) > 0 and models[0].get("task") == "chat":
            self.cache.set(model_id, True)
            return True

        self.cache.set(model_id, False)
        return False

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: str,
        logging_obj,
        optional_params: dict,
        litellm_params: dict,
        timeout: Union[float, httpx.Timeout],
        stream=False,
        acompletion=None,
        logger_fn=None,
        headers: dict = {},
    ) -> Union[ModelResponse, CustomStreamWrapper]:

        # we add stream not as an additional param, but as a primary prop on the request body
        if stream:
            # this is always defined if stream == True
            del optional_params["stream"]

        api_base = API_BASE
        # api_base = "http://localhost:8080/models/v2"

        headers = litellm.BytezChatConfig().validate_environment(
            api_key=api_key,
            headers=headers,
            messages=messages,
            optional_params=optional_params,
            model=model,
            litellm_params=litellm_params,
        )

        model_id = self.to_model_id(model)

        is_supported = self.validate_model_is_supported(model_id, headers)

        if not is_supported:
            raise Exception(f"Model: {model} does not support chat")

        completion_url = f"{api_base}/{model_id}"

        data = {
            "messages": messages,
            "stream": bool(stream),
            "params": optional_params,
        }

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
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
            if stream is True:
                return self.async_streaming(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=completion_url,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                    timeout=timeout,
                )  # type: ignore

            ### ASYNC COMPLETION
            return self.async_completion(
                model=model,
                messages=messages,
                data=data,
                api_base=completion_url,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                api_key=api_key,
                logging_obj=logging_obj,
                optional_params=optional_params,
                stream=False,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                headers=headers,
                timeout=timeout,
            )  # type: ignore

        ### SYNC STREAMING
        if stream is True:
            response = litellm.module_level_client.post(
                completion_url,
                headers=headers,
                data=json.dumps(data),
                stream=stream,
                timeout=timeout,  # type: ignore
            )
            _response = CustomStreamWrapper(
                response.iter_text(chunk_size=1),
                model,
                custom_llm_provider=CUSTOM_LLM_PROVIDER,
                logging_obj=logging_obj,
            )
            return _response

        ### SYNC COMPLETION
        response = litellm.module_level_client.post(
            url=completion_url,
            headers=headers,
            data=json.dumps(data),
            timeout=timeout,  # type: ignore
        )
        return self.process_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=optional_params.get("stream", False),
            logging_obj=logging_obj,  # type: ignore
            optional_params=optional_params,
            api_key=api_key,
            data=data,
            messages=messages,
            print_verbose=print_verbose,
            encoding=encoding,
        )

    async def async_completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        stream,
        data: dict,
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        litellm_params=None,
        logger_fn=None,
        headers={},
    ) -> ModelResponse:
        async_handler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.PREDIBASE,
            params={"timeout": timeout},
        )
        try:
            response = await async_handler.post(
                api_base, headers=headers, data=json.dumps(data)
            )
        except httpx.HTTPStatusError as e:
            raise BytezError(
                status_code=e.response.status_code,
                message="HTTPStatusError - received status_code={}, error_message={}".format(
                    e.response.status_code, e.response.text
                ),
            )
        except Exception as e:
            for exception in litellm.LITELLM_EXCEPTION_TYPES:
                if isinstance(e, exception):
                    raise e
            raise BytezError(
                status_code=500, message="{}".format(str(e))
            )  # don't use verbose_logger.exception, if exception is raised
        return self.process_response(
            model=model,
            response=response,
            model_response=model_response,
            stream=stream,
            logging_obj=logging_obj,
            api_key=api_key,
            data=data,
            messages=messages,
            print_verbose=print_verbose,
            optional_params=optional_params,
            encoding=encoding,
        )

    async def async_streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        data: dict,
        timeout: Union[float, httpx.Timeout],
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
    ) -> CustomStreamWrapper:
        data["stream"] = True

        streamwrapper = CustomStreamWrapper(
            completion_stream=None,
            make_call=partial(
                make_call,
                api_base=api_base,
                headers=headers,
                data=json.dumps(data),
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                timeout=timeout,
            ),
            model=model,
            custom_llm_provider=CUSTOM_LLM_PROVIDER,
            logging_obj=logging_obj,
        )
        return streamwrapper

    def embedding(self, *args, **kwargs):
        pass

    def process_response(  # noqa: PLR0915
        self,
        model: str,
        response: httpx.Response,
        model_response: ModelResponse,
        stream: bool,
        logging_obj: LiteLLMLoggingBaseClass,
        optional_params: dict,
        api_key: str,
        data: Union[dict, str],
        messages: list,
        print_verbose,
        encoding,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")
        ## RESPONSE OBJECT

        json = response.json()

        error = json.get("error")

        if error is not None:
            raise BytezError(
                message=str(json["error"]),
                status_code=response.status_code,
            )

        # meta data here
        model_response.created = int(time.time())
        model_response.model = model

        response_headers = response.headers

        # TODO usage

        # usage = Usage(
        #     prompt_tokens=prompt_tokens,
        #     completion_tokens=completion_tokens,
        #     total_tokens=total_tokens,
        # )
        # model_response.usage = usage  # type: ignore

        # TODO additional meta data such as inference time and model size, etc

        model_response._hidden_params["additional_headers"] = response_headers

        # Add the output
        output = json.get("output")

        message = model_response.choices[0].message  # type: ignore

        message.content = output

        # message.tool_calls
        # message.function_call
        # message.provider_specific_fields <-- this one is probably where we want to put our custom headers

        return model_response


async def make_call(
    client: AsyncHTTPHandler,
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    timeout: Optional[Union[float, httpx.Timeout]],
):
    response = await client.post(
        api_base, headers=headers, data=data, stream=True, timeout=timeout
    )

    if response.status_code != 200:
        raise BytezError(status_code=response.status_code, message=response.text)

    completion_stream = response.aiter_text(chunk_size=1)
    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=completion_stream,  # Pass the completion stream for logging
        additional_args={"complete_input_dict": data},
    )

    return completion_stream
