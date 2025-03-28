# What is this?
## handler file for TextCompletionCodestral Integration - https://codestral.com/

import json
from functools import partial
from typing import Callable, List, Optional, Union

import httpx  # type: ignore

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.prompt_templates.factory import (
    custom_prompt,
    prompt_factory,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)
from litellm.types.utils import TextChoices
from litellm.utils import CustomStreamWrapper, TextCompletionResponse


class TextCompletionCodestralError(Exception):
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
                url="https://docs.codestral.com/user-guide/inference/rest_api",
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
):
    response = await client.post(api_base, headers=headers, data=data, stream=True)

    if response.status_code != 200:
        raise TextCompletionCodestralError(
            status_code=response.status_code, message=response.text
        )

    completion_stream = response.aiter_lines()
    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=completion_stream,  # Pass the completion stream for logging
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class CodestralTextCompletion:
    def __init__(self) -> None:
        super().__init__()

    def _validate_environment(
        self,
        api_key: Optional[str],
        user_headers: dict,
    ) -> dict:
        if api_key is None:
            raise ValueError(
                "Missing CODESTRAL_API_Key - Please add CODESTRAL_API_Key to your environment variables"
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

    def process_text_completion_response(
        self,
        model: str,
        response: httpx.Response,
        model_response: TextCompletionResponse,
        stream: bool,
        logging_obj: LiteLLMLogging,
        optional_params: dict,
        api_key: str,
        data: Union[dict, str],
        messages: list,
        print_verbose,
        encoding,
    ) -> TextCompletionResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"codestral api: raw model_response: {response.text}")
        ## RESPONSE OBJECT
        if response.status_code != 200:
            raise TextCompletionCodestralError(
                message=str(response.text),
                status_code=response.status_code,
            )
        try:
            completion_response = response.json()
        except Exception:
            raise TextCompletionCodestralError(message=response.text, status_code=422)

        _original_choices = completion_response.get("choices", [])
        _choices: List[TextChoices] = []
        for choice in _original_choices:
            # This is what 1 choice looks like from codestral API
            # {
            #     "index": 0,
            #     "message": {
            #     "role": "assistant",
            #     "content": "\n assert is_odd(1)\n assert",
            #     "tool_calls": null
            #     },
            #     "finish_reason": "length",
            #     "logprobs": null
            #     }
            _finish_reason = None
            _index = 0
            _text = None
            _logprobs = None

            _choice_message = choice.get("message", {})
            _choice = litellm.utils.TextChoices(
                finish_reason=choice.get("finish_reason"),
                index=choice.get("index"),
                text=_choice_message.get("content"),
                logprobs=choice.get("logprobs"),
            )

            _choices.append(_choice)

        _response = litellm.TextCompletionResponse(
            id=completion_response.get("id"),
            choices=_choices,
            created=completion_response.get("created"),
            model=completion_response.get("model"),
            usage=completion_response.get("usage"),
            stream=False,
            object=completion_response.get("object"),
        )
        return _response

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: TextCompletionResponse,
        print_verbose: Callable,
        encoding,
        api_key: str,
        logging_obj,
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict = {},
    ) -> Union[TextCompletionResponse, CustomStreamWrapper]:
        headers = self._validate_environment(api_key, headers)

        if optional_params.pop("custom_endpoint", None) is True:
            completion_url = api_base
        else:
            completion_url = (
                api_base or "https://codestral.mistral.ai/v1/fim/completions"
            )

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
        config = litellm.CodestralTextCompletionConfig.get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        stream = optional_params.pop("stream", False)

        data = {
            "model": model,
            "prompt": prompt,
            **optional_params,
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
            response = litellm.module_level_client.post(
                completion_url,
                headers=headers,
                data=json.dumps(data),
                stream=stream,
            )
            _response = CustomStreamWrapper(
                response.iter_lines(),
                model,
                custom_llm_provider="codestral",
                logging_obj=logging_obj,
            )
            return _response
        ### SYNC COMPLETION
        else:

            response = litellm.module_level_client.post(
                url=completion_url,
                headers=headers,
                data=json.dumps(data),
            )
        return self.process_text_completion_response(
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
        model_response: TextCompletionResponse,
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
    ) -> TextCompletionResponse:

        async_handler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.TEXT_COMPLETION_CODESTRAL,
            params={"timeout": timeout},
        )
        try:

            response = await async_handler.post(
                api_base, headers=headers, data=json.dumps(data)
            )
        except httpx.HTTPStatusError as e:
            raise TextCompletionCodestralError(
                status_code=e.response.status_code,
                message="HTTPStatusError - {}".format(e.response.text),
            )
        except Exception as e:
            raise TextCompletionCodestralError(
                status_code=500, message="{}".format(str(e))
            )  # don't use verbose_logger.exception, if exception is raised
        return self.process_text_completion_response(
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
        model_response: TextCompletionResponse,
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
            ),
            model=model,
            custom_llm_provider="text-completion-codestral",
            logging_obj=logging_obj,
        )
        return streamwrapper

    def embedding(self, *args, **kwargs):
        pass
