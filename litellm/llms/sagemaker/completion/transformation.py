"""
Translate from OpenAI's `/v1/chat/completions` to Sagemaker's `/invoke`

In the Huggingface TGI format. 
"""

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from httpx._models import Headers, Response

import litellm
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.prompt_templates.factory import (
    custom_prompt,
    prompt_factory,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage

from ..common_utils import SagemakerError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class SagemakerConfig(BaseConfig):
    """
    Reference: https://d-uuwbxj1u4cnu.studio.us-west-2.sagemaker.aws/jupyter/default/lab/workspaces/auto-q/tree/DemoNotebooks/meta-textgeneration-llama-2-7b-SDK_1.ipynb
    """

    max_new_tokens: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    return_full_text: Optional[bool] = None

    def __init__(
        self,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        return_full_text: Optional[bool] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return SagemakerError(
            message=error_message, status_code=status_code, headers=headers
        )

    def get_supported_openai_params(self, model: str) -> List:
        return ["stream", "temperature", "max_tokens", "top_p", "stop", "n"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "temperature":
                if value == 0.0 or value == 0:
                    # hugging face exception raised when temp==0
                    # Failed: Error occurred: HuggingfaceException - Input validation error: `temperature` must be strictly positive
                    if not non_default_params.get(
                        "aws_sagemaker_allow_zero_temp", False
                    ):
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
            if param == "max_tokens":
                # HF TGI raises the following exception when max_new_tokens==0
                # Failed: Error occurred: HuggingfaceException - Input validation error: `max_new_tokens` must be strictly positive
                if value == 0:
                    value = 1
                optional_params["max_new_tokens"] = value
        non_default_params.pop("aws_sagemaker_allow_zero_temp", None)
        return optional_params

    def _transform_prompt(
        self,
        model: str,
        messages: List,
        custom_prompt_dict: dict,
        hf_model_name: Optional[str],
    ) -> str:
        if model in custom_prompt_dict:
            # check if the model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[model]
            prompt = custom_prompt(
                role_dict=model_prompt_details.get("roles", None),
                initial_prompt_value=model_prompt_details.get(
                    "initial_prompt_value", ""
                ),
                final_prompt_value=model_prompt_details.get("final_prompt_value", ""),
                messages=messages,
            )
        elif hf_model_name in custom_prompt_dict:
            # check if the base huggingface model has a registered custom prompt
            model_prompt_details = custom_prompt_dict[hf_model_name]
            prompt = custom_prompt(
                role_dict=model_prompt_details.get("roles", None),
                initial_prompt_value=model_prompt_details.get(
                    "initial_prompt_value", ""
                ),
                final_prompt_value=model_prompt_details.get("final_prompt_value", ""),
                messages=messages,
            )
        else:
            if hf_model_name is None:
                if "llama-2" in model.lower():  # llama-2 model
                    if "chat" in model.lower():  # apply llama2 chat template
                        hf_model_name = "meta-llama/Llama-2-7b-chat-hf"
                    else:  # apply regular llama2 template
                        hf_model_name = "meta-llama/Llama-2-7b"
            hf_model_name = (
                hf_model_name or model
            )  # pass in hf model name for pulling it's prompt template - (e.g. `hf_model_name="meta-llama/Llama-2-7b-chat-hf` applies the llama2 chat template to the prompt)
            prompt: str = prompt_factory(model=hf_model_name, messages=messages)  # type: ignore

        return prompt

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        inference_params = optional_params.copy()
        stream = inference_params.pop("stream", False)
        data: Dict = {"parameters": inference_params}
        if stream is True:
            data["stream"] = True

        custom_prompt_dict = (
            litellm_params.get("custom_prompt_dict", None) or litellm.custom_prompt_dict
        )

        hf_model_name = litellm_params.get("hf_model_name", None)

        prompt = self._transform_prompt(
            model=model,
            messages=messages,
            custom_prompt_dict=custom_prompt_dict,
            hf_model_name=hf_model_name,
        )
        data["inputs"] = prompt

        return data

    async def async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return await asyncify(self.transform_request)(
            model, messages, optional_params, litellm_params, headers
        )

    def transform_response(
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: str,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        completion_response = raw_response.json()
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=completion_response,
            additional_args={"complete_input_dict": request_data},
        )

        prompt = request_data["inputs"]

        ## RESPONSE OBJECT
        try:
            if isinstance(completion_response, list):
                completion_response_choices = completion_response[0]
            else:
                completion_response_choices = completion_response
            completion_output = ""
            if "generation" in completion_response_choices:
                completion_output += completion_response_choices["generation"]
            elif "generated_text" in completion_response_choices:
                completion_output += completion_response_choices["generated_text"]

            # check if the prompt template is part of output, if so - filter it out
            if completion_output.startswith(prompt) and "<s>" in prompt:
                completion_output = completion_output.replace(prompt, "", 1)

            model_response.choices[0].message.content = completion_output  # type: ignore
        except Exception:
            raise SagemakerError(
                message=f"LiteLLM Error: Unable to parse sagemaker RAW RESPONSE {json.dumps(completion_response)}",
                status_code=500,
            )

        ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here.
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content", ""))
        )

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response

    def validate_environment(
        self,
        headers: Optional[dict],
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        headers = {"Content-Type": "application/json"}

        if headers is not None:
            headers = {"Content-Type": "application/json", **headers}

        return headers
