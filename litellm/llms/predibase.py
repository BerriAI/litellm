# What is this?
## Controller file for Predibase Integration - https://predibase.com/

import copy
import json
import os
import time
import traceback
import types
from enum import Enum
from functools import partial
from typing import Callable, List, Literal, Optional, Union

import httpx  # type: ignore
import requests  # type: ignore

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
from litellm import verbose_logger
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.utils import Choices, CustomStreamWrapper, Message, ModelResponse, Usage

from .base import BaseLLM
from .prompt_templates.factory import custom_prompt, prompt_factory


class PredibaseError(Exception):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
    ):
        self.status_code = status_code
        self.message = message
        if request is not None:
            self.request = request
        else:
            self.request = httpx.Request(
                method="POST",
                url="https://docs.predibase.com/user-guide/inference/rest_api",
            )
        if response is not None:
            self.response = response
        else:
            self.response = httpx.Response(
                status_code=status_code, request=self.request
            )
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


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
        raise PredibaseError(status_code=response.status_code, message=response.text)

    completion_stream = response.aiter_lines()
    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=completion_stream,  # Pass the completion stream for logging
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class PredibaseConfig:
    """
    Reference:  https://docs.predibase.com/user-guide/inference/rest_api

    """

    adapter_id: Optional[str] = None
    adapter_source: Optional[Literal["pbase", "hub", "s3"]] = None
    best_of: Optional[int] = None
    decoder_input_details: Optional[bool] = None
    details: bool = True  # enables returning logprobs + best of
    max_new_tokens: int = (
        256  # openai default - requests hang if max_new_tokens not given
    )
    repetition_penalty: Optional[float] = None
    return_full_text: Optional[bool] = (
        False  # by default don't return the input as part of the output
    )
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[int] = None
    truncate: Optional[int] = None
    typical_p: Optional[float] = None
    watermark: Optional[bool] = None

    def __init__(
        self,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        details: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self):
        return [
            "stream",
            "temperature",
            "max_completion_tokens",
            "max_tokens",
            "top_p",
            "stop",
            "n",
            "response_format",
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            # temperature, top_p, n, stream, stop, max_tokens, n, presence_penalty default to None
            if param == "temperature":
                if value == 0.0 or value == 0:
                    # hugging face exception raised when temp==0
                    # Failed: Error occurred: HuggingfaceException - Input validation error: `temperature` must be strictly positive
                    value = 0.01
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "n":
                optional_params["best_of"] = value
                optional_params["do_sample"] = (
                    True  # Need to sample if you want best of for hf inference endpoints
                )
            if param == "stream":
                optional_params["stream"] = value
            if param == "stop":
                optional_params["stop"] = value
            if param == "max_tokens" or param == "max_completion_tokens":
                # HF TGI raises the following exception when max_new_tokens==0
                # Failed: Error occurred: HuggingfaceException - Input validation error: `max_new_tokens` must be strictly positive
                if value == 0:
                    value = 1
                optional_params["max_new_tokens"] = value
            if param == "echo":
                # https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceClient.text_generation.decoder_input_details
                #  Return the decoder input token logprobs and ids. You must set details=True as well for it to be taken into account. Defaults to False
                optional_params["decoder_input_details"] = True
            if param == "response_format":
                optional_params["response_format"] = value
        return optional_params


class PredibaseChatCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def _validate_environment(
        self, api_key: Optional[str], user_headers: dict, tenant_id: Optional[str]
    ) -> dict:
        if api_key is None:
            raise ValueError(
                "Missing Predibase API Key - A call is being made to predibase but no key is set either in the environment variables or via params"
            )
        if tenant_id is None:
            raise ValueError(
                "Missing Predibase Tenant ID - Required for making the request. Set dynamically (e.g. `completion(..tenant_id=<MY-ID>)`) or in env - `PREDIBASE_TENANT_ID`."
            )
        headers = {
            "content-type": "application/json",
            "Authorization": "Bearer {}".format(api_key),
        }
        if user_headers is not None and isinstance(user_headers, dict):
            headers = {**headers, **user_headers}
        return headers

    def output_parser(self, generated_text: str):
        """
        Parse the output text to remove any special characters. In our current approach we just check for ChatML tokens.

        Initial issue that prompted this - https://github.com/BerriAI/litellm/issues/763
        """
        chat_template_tokens = [
            "<|assistant|>",
            "<|system|>",
            "<|user|>",
            "<s>",
            "</s>",
        ]
        for token in chat_template_tokens:
            if generated_text.strip().startswith(token):
                generated_text = generated_text.replace(token, "", 1)
            if generated_text.endswith(token):
                generated_text = generated_text[::-1].replace(token[::-1], "", 1)[::-1]
        return generated_text

    def process_response(
        self,
        model: str,
        response: Union[requests.Response, httpx.Response],
        model_response: ModelResponse,
        stream: bool,
        logging_obj: litellm.litellm_core_utils.litellm_logging.Logging,
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
        try:
            completion_response = response.json()
        except:
            raise PredibaseError(message=response.text, status_code=422)
        if "error" in completion_response:
            raise PredibaseError(
                message=str(completion_response["error"]),
                status_code=response.status_code,
            )
        else:
            if not isinstance(completion_response, dict):
                raise PredibaseError(
                    status_code=422,
                    message=f"'completion_response' is not a dictionary - {completion_response}",
                )
            elif "generated_text" not in completion_response:
                raise PredibaseError(
                    status_code=422,
                    message=f"'generated_text' is not a key response dictionary - {completion_response}",
                )
            if len(completion_response["generated_text"]) > 0:
                model_response.choices[0].message.content = self.output_parser(  # type: ignore
                    completion_response["generated_text"]
                )
            ## GETTING LOGPROBS + FINISH REASON
            if (
                "details" in completion_response
                and "tokens" in completion_response["details"]
            ):
                model_response.choices[0].finish_reason = map_finish_reason(
                    completion_response["details"]["finish_reason"]
                )
                sum_logprob = 0
                for token in completion_response["details"]["tokens"]:
                    if token["logprob"] is not None:
                        sum_logprob += token["logprob"]
                setattr(
                    model_response.choices[0].message,  # type: ignore
                    "_logprob",
                    sum_logprob,  # [TODO] move this to using the actual logprobs
                )
            if "best_of" in optional_params and optional_params["best_of"] > 1:
                if (
                    "details" in completion_response
                    and "best_of_sequences" in completion_response["details"]
                ):
                    choices_list = []
                    for idx, item in enumerate(
                        completion_response["details"]["best_of_sequences"]
                    ):
                        sum_logprob = 0
                        for token in item["tokens"]:
                            if token["logprob"] is not None:
                                sum_logprob += token["logprob"]
                        if len(item["generated_text"]) > 0:
                            message_obj = Message(
                                content=self.output_parser(item["generated_text"]),
                                logprobs=sum_logprob,
                            )
                        else:
                            message_obj = Message(content=None)
                        choice_obj = Choices(
                            finish_reason=map_finish_reason(item["finish_reason"]),
                            index=idx + 1,
                            message=message_obj,
                        )
                        choices_list.append(choice_obj)
                    model_response.choices.extend(choices_list)

        ## CALCULATING USAGE
        prompt_tokens = 0
        try:
            prompt_tokens = litellm.token_counter(messages=messages)
        except Exception:
            # this should remain non blocking we should not block a response returning if calculating usage fails
            pass
        output_text = model_response["choices"][0]["message"].get("content", "")
        if output_text is not None and len(output_text) > 0:
            completion_tokens = 0
            try:
                completion_tokens = len(
                    encoding.encode(
                        model_response["choices"][0]["message"].get("content", "")
                    )
                )  ##[TODO] use a model-specific tokenizer
            except:
                # this should remain non blocking we should not block a response returning if calculating usage fails
                pass
        else:
            completion_tokens = 0

        total_tokens = prompt_tokens + completion_tokens

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        model_response.usage = usage  # type: ignore

        ## RESPONSE HEADERS
        predibase_headers = response.headers
        response_headers = {}
        for k, v in predibase_headers.items():
            if k.startswith("x-"):
                response_headers["llm_provider-{}".format(k)] = v

        model_response._hidden_params["additional_headers"] = response_headers

        return model_response

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: str,
        logging_obj,
        optional_params: dict,
        tenant_id: str,
        timeout: Union[float, httpx.Timeout],
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict = {},
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        headers = self._validate_environment(api_key, headers, tenant_id=tenant_id)
        completion_url = ""
        input_text = ""
        base_url = "https://serving.app.predibase.com"

        if "https" in model:
            completion_url = model
        elif api_base:
            base_url = api_base
        elif "PREDIBASE_API_BASE" in os.environ:
            base_url = os.getenv("PREDIBASE_API_BASE", "")

        completion_url = f"{base_url}/{tenant_id}/deployments/v2/llms/{model}"

        if optional_params.get("stream", False) is True:
            completion_url += "/generate_stream"
        else:
            completion_url += "/generate"

        if model in custom_prompt_dict:
            # check if the model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details["roles"],
                initial_prompt_value=model_prompt_details["initial_prompt_value"],
                final_prompt_value=model_prompt_details["final_prompt_value"],
                messages=messages,
            )
        else:
            prompt = prompt_factory(model=model, messages=messages)

        ## Load Config
        config = litellm.PredibaseConfig.get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        stream = optional_params.pop("stream", False)

        data = {
            "inputs": prompt,
            "parameters": optional_params,
        }
        input_text = prompt
        ## LOGGING
        logging_obj.pre_call(
            input=input_text,
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
            else:
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
            response = requests.post(
                completion_url,
                headers=headers,
                data=json.dumps(data),
                stream=stream,
                timeout=timeout,  # type: ignore
            )
            _response = CustomStreamWrapper(
                response.iter_lines(),
                model,
                custom_llm_provider="predibase",
                logging_obj=logging_obj,
            )
            return _response
        ### SYNC COMPLETION
        else:
            response = requests.post(
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

        async_handler = AsyncHTTPHandler(timeout=httpx.Timeout(timeout=timeout))
        try:
            response = await async_handler.post(
                api_base, headers=headers, data=json.dumps(data)
            )
        except httpx.HTTPStatusError as e:
            raise PredibaseError(
                status_code=e.response.status_code,
                message="HTTPStatusError - received status_code={}, error_message={}".format(
                    e.response.status_code, e.response.text
                ),
            )
        except Exception as e:
            for exception in litellm.LITELLM_EXCEPTION_TYPES:
                if isinstance(e, exception):
                    raise e
            raise PredibaseError(
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
            custom_llm_provider="predibase",
            logging_obj=logging_obj,
        )
        return streamwrapper

    def embedding(self, *args, **kwargs):
        pass
