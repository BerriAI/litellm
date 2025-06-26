# What is this?
## Controller file for Predibase Integration - https://predibase.com/
from typing import Dict
import json
import time
from functools import partial
from typing import Optional, Union

import httpx  # type: ignore
from functools import lru_cache

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)
from litellm.types.utils import LiteLLMLoggingBaseClass
from litellm.utils import CustomStreamWrapper, ModelResponse

from ..common_utils import BytezError, validate_environment
from ..openai_to_bytez_param_map import map_openai_params_to_bytez_params

CUSTOM_LLM_PROVIDER = "bytez"

API_BASE = "https://api.bytez.com/models/v2"


class BytezChatCompletion:
    def __init__(self) -> None:
        super().__init__()

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        api_key: str,
        logging_obj,
        optional_params: dict,
        litellm_params: dict,
        timeout: Union[float, httpx.Timeout],
        stream=False,
        acompletion=None,
        headers: dict = {},
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        # throws if there is a problem
        validate_environment(api_key=api_key, messages=messages)

        self.update_headers(headers, api_key)

        # this will remap openai params if we can, otherwise it will throw
        optional_params = map_openai_params_to_bytez_params(optional_params)

        # we add stream not as an additional param, but as a primary prop on the request body, this is always defined if stream == True
        if stream:
            del optional_params["stream"]

        model_id = self.to_model_id(model)

        is_supported = self.validate_model_is_supported(model_id, headers)

        if not is_supported:
            raise Exception(f"Model: {model} does not support chat")

        completion_url = f"{API_BASE}/{model_id}"

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
                    logging_obj=logging_obj,
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
                api_key=api_key,
                logging_obj=logging_obj,
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
            logging_obj=logging_obj,  # type: ignore
            api_key=api_key,
            data=data,
            messages=messages,
        )

    def to_model_id(self, model):
        # remove bytez off of the front of the model
        model = "/".join(model.split("/")[1:])
        return model

    def validate_model_is_supported(self, model_id: str, headers: Dict) -> bool:
        headers_tuple = tuple(headers.items())
        return self._validate_model_is_supported_cached(model_id, headers_tuple)

    @lru_cache(maxsize=128)
    def _validate_model_is_supported_cached(
        self, model_id: str, headers_tuple: tuple
    ) -> bool:
        headers = dict(headers_tuple)

        url = f"{API_BASE}/list/models?modelId={model_id}"

        response = httpx.request(method="GET", url=url, headers=headers)
        response_data = response.json()

        error = response_data.get("error")
        if error:
            raise Exception(error)

        models = response_data.get("output", [])
        is_supported = len(models) > 0 and models[0].get("task") == "chat"

        return is_supported

    def update_headers(self, headers: dict, api_key: str) -> None:
        headers.update(
            {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Key {api_key}",
                "User-Agent": "litellm-client",
            }
        )

    async def async_completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        api_key,
        logging_obj,
        data: dict,
        timeout: Union[float, httpx.Timeout],
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
            logging_obj=logging_obj,
            api_key=api_key,
            data=data,
            messages=messages,
        )

    async def async_streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        logging_obj,
        data: dict,
        timeout: Union[float, httpx.Timeout],
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
                messages=messages,
                logging_obj=logging_obj,
                timeout=timeout,
            ),
            model=model,
            custom_llm_provider=CUSTOM_LLM_PROVIDER,
            logging_obj=logging_obj,
        )
        return streamwrapper

    def process_response(
        self,
        model: str,
        response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingBaseClass,
        api_key: str,
        data: Union[dict, str],
        messages: list,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
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
