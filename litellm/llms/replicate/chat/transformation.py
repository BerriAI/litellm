from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

import litellm
from litellm.constants import REPLICATE_MODEL_NAME_WITH_ID_LENGTH
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    custom_prompt,
    prompt_factory,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage
from litellm.utils import token_counter

from ..common_utils import ReplicateError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class ReplicateConfig(BaseConfig):
    """
    Reference: https://replicate.com/meta/llama-2-70b-chat/api
    - `prompt` (string): The prompt to send to the model.

    - `system_prompt` (string): The system prompt to send to the model. This is prepended to the prompt and helps guide system behavior. Default value: `You are a helpful assistant`.

    - `max_new_tokens` (integer): Maximum number of tokens to generate. Typically, a word is made up of 2-3 tokens. Default value: `128`.

    - `min_new_tokens` (integer): Minimum number of tokens to generate. To disable, set to `-1`. A word is usually 2-3 tokens. Default value: `-1`.

    - `temperature` (number): Adjusts the randomness of outputs. Values greater than 1 increase randomness, 0 is deterministic, and 0.75 is a reasonable starting value. Default value: `0.75`.

    - `top_p` (number): During text decoding, it samples from the top `p` percentage of most likely tokens. Reduce this to ignore less probable tokens. Default value: `0.9`.

    - `top_k` (integer): During text decoding, samples from the top `k` most likely tokens. Reduce this to ignore less probable tokens. Default value: `50`.

    - `stop_sequences` (string): A comma-separated list of sequences to stop generation at. For example, inputting '<end>,<stop>' will cease generation at the first occurrence of either 'end' or '<stop>'.

    - `seed` (integer): This is the seed for the random generator. Leave it blank to randomize the seed.

    - `debug` (boolean): If set to `True`, it provides debugging output in logs.

    Please note that Replicate's mapping of these parameters can be inconsistent across different models, indicating that not all of these parameters may be available for use with all models.
    """

    system_prompt: Optional[str] = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[str] = None
    seed: Optional[int] = None
    debug: Optional[bool] = None

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[str] = None,
        seed: Optional[int] = None,
        debug: Optional[bool] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "seed",
            "tools",
            "tool_choice",
            "functions",
            "function_call",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "stream":
                optional_params["stream"] = value
            if param == "max_tokens":
                if "vicuna" in model or "flan" in model:
                    optional_params["max_length"] = value
                elif "meta/codellama-13b" in model:
                    optional_params["max_tokens"] = value
                else:
                    optional_params["max_new_tokens"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "stop":
                optional_params["stop_sequences"] = value

        return optional_params

    # Function to extract version ID from model string
    def model_to_version_id(self, model: str) -> str:
        if ":" in model:
            split_model = model.split(":")
            return split_model[1]
        return model

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return ReplicateError(
            status_code=status_code, message=error_message, headers=headers
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        version_id = self.model_to_version_id(model)
        base_url = api_base
        if "deployments" in version_id:
            version_id = version_id.replace("deployments/", "")
            base_url = f"https://api.replicate.com/v1/deployments/{version_id}"
        else:  # assume it's a model
            base_url = f"https://api.replicate.com/v1/models/{version_id}"

        base_url = f"{base_url}/predictions"
        return base_url

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        ## Load Config
        config = litellm.ReplicateConfig.get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > replicate_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        system_prompt = None
        if optional_params is not None and "supports_system_prompt" in optional_params:
            supports_sys_prompt = optional_params.pop("supports_system_prompt")
        else:
            supports_sys_prompt = False

        if supports_sys_prompt:
            for i in range(len(messages)):
                if messages[i]["role"] == "system":
                    first_sys_message = messages.pop(i)
                    system_prompt = convert_content_list_to_str(first_sys_message)
                    break

        if model in litellm.custom_prompt_dict:
            # check if the model has a registered custom prompt
            model_prompt_details = litellm.custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details.get("roles", {}),
                initial_prompt_value=model_prompt_details.get(
                    "initial_prompt_value", ""
                ),
                final_prompt_value=model_prompt_details.get("final_prompt_value", ""),
                bos_token=model_prompt_details.get("bos_token", ""),
                eos_token=model_prompt_details.get("eos_token", ""),
                messages=messages,
            )
        else:
            prompt = prompt_factory(model=model, messages=messages)

        if prompt is None or not isinstance(prompt, str):
            raise ReplicateError(
                status_code=400,
                message="LiteLLM Error - prompt is not a string - {}".format(prompt),
                headers={},
            )

        # If system prompt is supported, and a system prompt is provided, use it
        if system_prompt is not None:
            input_data = {
                "prompt": prompt,
                "system_prompt": system_prompt,
                **optional_params,
            }
        # Otherwise, use the prompt as is
        else:
            input_data = {"prompt": prompt, **optional_params}

        version_id = self.model_to_version_id(model)
        request_data: dict = {"input": input_data}
        if ":" in version_id and len(version_id) > REPLICATE_MODEL_NAME_WITH_ID_LENGTH:
            model_parts = version_id.split(":")
            if (
                len(model_parts) > 1
                and len(model_parts[1]) == REPLICATE_MODEL_NAME_WITH_ID_LENGTH
            ):  ## checks if model name has a 64 digit code - e.g. "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
                request_data["version"] = model_parts[1]

        return request_data

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
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
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )
        raw_response_json = raw_response.json()
        if raw_response_json.get("status") != "succeeded":
            raise ReplicateError(
                status_code=422,
                message="LiteLLM Error - prediction not succeeded - {}".format(
                    raw_response_json
                ),
                headers=raw_response.headers,
            )
        outputs = raw_response_json.get("output", [])
        response_str = "".join(outputs)
        if len(response_str) == 0:  # edge case, where result from replicate is empty
            response_str = " "

        ## Building RESPONSE OBJECT
        if len(response_str) >= 1:
            model_response.choices[0].message.content = response_str  # type: ignore

        # Calculate usage
        prompt_tokens = token_counter(model=model, messages=messages)
        completion_tokens = token_counter(
            model=model,
            text=response_str,
            count_response_tokens=True,
        )
        model_response.model = "replicate/" + model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)

        return model_response

    def get_prediction_url(self, response: httpx.Response) -> str:
        """
        response json: {
        ...,
        "urls":{"cancel":"https://api.replicate.com/v1/predictions/gqsmqmp1pdrj00cknr08dgmvb4/cancel","get":"https://api.replicate.com/v1/predictions/gqsmqmp1pdrj00cknr08dgmvb4","stream":"https://stream-b.svc.rno2.c.replicate.net/v1/streams/eot4gbydowuin4snhncydwxt57dfwgsc3w3snycx5nid7oef7jga"}
        }
        """
        response_json = response.json()
        prediction_url = response_json.get("urls", {}).get("get")
        if prediction_url is None:
            raise ReplicateError(
                status_code=400,
                message="LiteLLM Error - prediction url is None - {}".format(
                    response_json
                ),
                headers=response.headers,
            )
        return prediction_url

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
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }
        return headers
