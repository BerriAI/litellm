import os
import time
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union

from httpx import Headers, Response

import litellm
from litellm.constants import DEFAULT_MAX_TOKENS
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.prompt_templates.factory import (
    custom_prompt,
    prompt_factory,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from ..common_utils import PredibaseError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class PredibaseConfig(BaseConfig):
    """
    Reference:  https://docs.predibase.com/user-guide/inference/rest_api
    """

    adapter_id: Optional[str] = None
    adapter_source: Optional[Literal["pbase", "hub", "s3"]] = None
    best_of: Optional[int] = None
    decoder_input_details: Optional[bool] = None
    details: bool = True  # enables returning logprobs + best of
    max_new_tokens: int = (
        DEFAULT_MAX_TOKENS  # openai default - requests hang if max_new_tokens not given
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
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str):
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

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
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

    def transform_response(  # noqa: PLR0915
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        logging_obj.post_call(
            input=messages,
            api_key=api_key or "",
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )
        try:
            completion_response = raw_response.json()
        except Exception:
            raise PredibaseError(message=raw_response.text, status_code=422)

        if "error" in completion_response:
            raise PredibaseError(
                message=str(completion_response["error"]),
                status_code=raw_response.status_code,
            )
        elif not isinstance(completion_response, dict):
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

        effective_best_of = optional_params.get("best_of")
        if effective_best_of is None:
            effective_best_of = request_data.get("parameters", {}).get("best_of", 0)
        try:
            best_of_value = int(effective_best_of)
        except (TypeError, ValueError):
            best_of_value = 0

        if best_of_value > 1:
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

        prompt_tokens = 0
        try:
            prompt_tokens = litellm.token_counter(messages=messages)
        except Exception:
            # Keep usage calculation non-blocking if token counting fails.
            pass
        output_text = model_response["choices"][0]["message"].get("content", "")
        if output_text is not None and len(output_text) > 0:
            completion_tokens = 0
            try:
                completion_tokens = len(
                    encoding.encode(
                        model_response["choices"][0]["message"].get("content", "")
                    )
                )
            except Exception:
                # Keep usage calculation non-blocking if encoding fails.
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

        predibase_headers = raw_response.headers
        response_headers = {}
        for k, v in predibase_headers.items():
            if k.startswith("x-"):
                response_headers[f"llm_provider-{k}"] = v

        model_response._hidden_params["additional_headers"] = response_headers

        return model_response

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        custom_prompt_dict = litellm_params.get("custom_prompt_dict", {})
        if model in custom_prompt_dict:
            model_prompt_details = custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details["roles"],
                initial_prompt_value=model_prompt_details["initial_prompt_value"],
                final_prompt_value=model_prompt_details["final_prompt_value"],
                messages=messages,
            )
        else:
            prompt = prompt_factory(model=model, messages=messages)

        request_optional_params = {**optional_params}
        config = self.get_config()
        for k, v in config.items():
            if k not in request_optional_params:
                request_optional_params[k] = v

        request_optional_params.pop("stream", None)
        return {
            "inputs": prompt,
            "parameters": request_optional_params,
        }

    @staticmethod
    def output_parser(generated_text: str) -> str:
        """
        Parse the output text to remove any special characters.

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

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        tenant_id = litellm_params.get("predibase_tenant_id") or litellm_params.get(
            "tenant_id"
        )
        if tenant_id is None:
            raise ValueError(
                "Missing Predibase Tenant ID - Required for making the request. Set dynamically (e.g. `completion(..tenant_id=<MY-ID>)`) or in env - `PREDIBASE_TENANT_ID`."
            )

        base_url = "https://serving.app.predibase.com"
        if api_base:
            base_url = api_base
        elif "PREDIBASE_API_BASE" in os.environ:
            base_url = os.getenv("PREDIBASE_API_BASE", "")

        completion_url = f"{base_url}/{tenant_id}/deployments/v2/llms/{model}"
        should_stream = (
            stream if stream is not None else optional_params.get("stream", False)
        )
        if should_stream is True:
            completion_url += "/generate_stream"
        else:
            completion_url += "/generate"
        return completion_url

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return PredibaseError(
            status_code=status_code, message=error_message, headers=headers
        )

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
        if api_key is None:
            raise ValueError(
                "Missing Predibase API Key - A call is being made to predibase but no key is set either in the environment variables or via params"
            )

        default_headers = {
            "content-type": "application/json",
            "Authorization": "Bearer {}".format(api_key),
        }
        if headers is not None and isinstance(headers, dict):
            headers = {**default_headers, **headers}
        return headers
