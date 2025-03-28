import json
import os
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    custom_prompt,
    prompt_factory,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from litellm.utils import token_counter

from ..common_utils import HuggingfaceError, hf_task_list, hf_tasks, output_parser

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


tgi_models_cache = None
conv_models_cache = None


class HuggingfaceChatConfig(BaseConfig):
    """
    Reference: https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/compat_generate
    """

    hf_task: Optional[hf_tasks] = (
        None  # litellm-specific param, used to know the api spec to use when calling huggingface api
    )
    best_of: Optional[int] = None
    decoder_input_details: Optional[bool] = None
    details: Optional[bool] = True  # enables returning logprobs + best of
    max_new_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    return_full_text: Optional[bool] = (
        False  # by default don't return the input as part of the output
    )
    seed: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_n_tokens: Optional[int] = None
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
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
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

    def get_special_options_params(self):
        return ["use_cache", "wait_for_model"]

    def get_supported_openai_params(self, model: str):
        return [
            "stream",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "stop",
            "n",
            "echo",
        ]

    def map_openai_params(
        self,
        non_default_params: Dict,
        optional_params: Dict,
        model: str,
        drop_params: bool,
    ) -> Dict:
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

        return optional_params

    def get_hf_api_key(self) -> Optional[str]:
        return get_secret_str("HUGGINGFACE_API_KEY")

    def read_tgi_conv_models(self):
        try:
            global tgi_models_cache, conv_models_cache
            # Check if the cache is already populated
            # so we don't keep on reading txt file if there are 1k requests
            if (tgi_models_cache is not None) and (conv_models_cache is not None):
                return tgi_models_cache, conv_models_cache
            # If not, read the file and populate the cache
            tgi_models = set()
            script_directory = os.path.dirname(os.path.abspath(__file__))
            script_directory = os.path.dirname(script_directory)
            # Construct the file path relative to the script's directory
            file_path = os.path.join(
                script_directory,
                "huggingface_llms_metadata",
                "hf_text_generation_models.txt",
            )

            with open(file_path, "r") as file:
                for line in file:
                    tgi_models.add(line.strip())

            # Cache the set for future use
            tgi_models_cache = tgi_models

            # If not, read the file and populate the cache
            file_path = os.path.join(
                script_directory,
                "huggingface_llms_metadata",
                "hf_conversational_models.txt",
            )
            conv_models = set()
            with open(file_path, "r") as file:
                for line in file:
                    conv_models.add(line.strip())
            # Cache the set for future use
            conv_models_cache = conv_models
            return tgi_models, conv_models
        except Exception:
            return set(), set()

    def get_hf_task_for_model(self, model: str) -> Tuple[hf_tasks, str]:
        # read text file, cast it to set
        # read the file called "huggingface_llms_metadata/hf_text_generation_models.txt"
        if model.split("/")[0] in hf_task_list:
            split_model = model.split("/", 1)
            return split_model[0], split_model[1]  # type: ignore
        tgi_models, conversational_models = self.read_tgi_conv_models()

        if model in tgi_models:
            return "text-generation-inference", model
        elif model in conversational_models:
            return "conversational", model
        elif "roneneldan/TinyStories" in model:
            return "text-generation", model
        else:
            return "text-generation-inference", model  # default to tgi

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        task = litellm_params.get("task", None)
        ## VALIDATE API FORMAT
        if task is None or not isinstance(task, str) or task not in hf_task_list:
            raise Exception(
                "Invalid hf task - {}. Valid formats - {}.".format(task, hf_tasks)
            )

        ## Load Config
        config = litellm.HuggingfaceConfig.get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > huggingfaceConfig(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        ### MAP INPUT PARAMS
        #### HANDLE SPECIAL PARAMS
        special_params = self.get_special_options_params()
        special_params_dict = {}
        # Create a list of keys to pop after iteration
        keys_to_pop = []

        for k, v in optional_params.items():
            if k in special_params:
                special_params_dict[k] = v
                keys_to_pop.append(k)

        # Pop the keys from the dictionary after iteration
        for k in keys_to_pop:
            optional_params.pop(k)
        if task == "conversational":
            inference_params = deepcopy(optional_params)
            inference_params.pop("details")
            inference_params.pop("return_full_text")
            past_user_inputs = []
            generated_responses = []
            text = ""
            for message in messages:
                if message["role"] == "user":
                    if text != "":
                        past_user_inputs.append(text)
                    text = convert_content_list_to_str(message)
                elif message["role"] == "assistant" or message["role"] == "system":
                    generated_responses.append(convert_content_list_to_str(message))
            data = {
                "inputs": {
                    "text": text,
                    "past_user_inputs": past_user_inputs,
                    "generated_responses": generated_responses,
                },
                "parameters": inference_params,
            }

        elif task == "text-generation-inference":
            # always send "details" and "return_full_text" as params
            if model in litellm.custom_prompt_dict:
                # check if the model has a registered custom prompt
                model_prompt_details = litellm.custom_prompt_dict[model]
                prompt = custom_prompt(
                    role_dict=model_prompt_details.get("roles", None),
                    initial_prompt_value=model_prompt_details.get(
                        "initial_prompt_value", ""
                    ),
                    final_prompt_value=model_prompt_details.get(
                        "final_prompt_value", ""
                    ),
                    messages=messages,
                )
            else:
                prompt = prompt_factory(model=model, messages=messages)
            data = {
                "inputs": prompt,  # type: ignore
                "parameters": optional_params,
                "stream": (  # type: ignore
                    True
                    if "stream" in optional_params
                    and isinstance(optional_params["stream"], bool)
                    and optional_params["stream"] is True  # type: ignore
                    else False
                ),
            }
        else:
            # Non TGI and Conversational llms
            # We need this branch, it removes 'details' and 'return_full_text' from params
            if model in litellm.custom_prompt_dict:
                # check if the model has a registered custom prompt
                model_prompt_details = litellm.custom_prompt_dict[model]
                prompt = custom_prompt(
                    role_dict=model_prompt_details.get("roles", {}),
                    initial_prompt_value=model_prompt_details.get(
                        "initial_prompt_value", ""
                    ),
                    final_prompt_value=model_prompt_details.get(
                        "final_prompt_value", ""
                    ),
                    bos_token=model_prompt_details.get("bos_token", ""),
                    eos_token=model_prompt_details.get("eos_token", ""),
                    messages=messages,
                )
            else:
                prompt = prompt_factory(model=model, messages=messages)
            inference_params = deepcopy(optional_params)
            inference_params.pop("details")
            inference_params.pop("return_full_text")
            data = {
                "inputs": prompt,  # type: ignore
            }
            if task == "text-generation-inference":
                data["parameters"] = inference_params
                data["stream"] = (  # type: ignore
                    True  # type: ignore
                    if "stream" in optional_params and optional_params["stream"] is True
                    else False
                )

        ### RE-ADD SPECIAL PARAMS
        if len(special_params_dict.keys()) > 0:
            data.update({"options": special_params_dict})

        return data

    def get_api_base(self, api_base: Optional[str], model: str) -> str:
        """
        Get the API base for the Huggingface API.

        Do not add the chat/embedding/rerank extension here. Let the handler do this.
        """
        if "https" in model:
            completion_url = model
        elif api_base is not None:
            completion_url = api_base
        elif "HF_API_BASE" in os.environ:
            completion_url = os.getenv("HF_API_BASE", "")
        elif "HUGGINGFACE_API_BASE" in os.environ:
            completion_url = os.getenv("HUGGINGFACE_API_BASE", "")
        else:
            completion_url = f"https://api-inference.huggingface.co/models/{model}"

        return completion_url

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        default_headers = {
            "content-type": "application/json",
        }
        if api_key is not None:
            default_headers["Authorization"] = (
                f"Bearer {api_key}"  # Huggingface Inference Endpoint default is to accept bearer tokens
            )

        headers = {**headers, **default_headers}
        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return HuggingfaceError(
            status_code=status_code, message=error_message, headers=headers
        )

    def _convert_streamed_response_to_complete_response(
        self,
        response: httpx.Response,
        logging_obj: LoggingClass,
        model: str,
        data: dict,
        api_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        streamed_response = CustomStreamWrapper(
            completion_stream=response.iter_lines(),
            model=model,
            custom_llm_provider="huggingface",
            logging_obj=logging_obj,
        )
        content = ""
        for chunk in streamed_response:
            content += chunk["choices"][0]["delta"]["content"]
        completion_response: List[Dict[str, Any]] = [{"generated_text": content}]
        ## LOGGING
        logging_obj.post_call(
            input=data,
            api_key=api_key,
            original_response=completion_response,
            additional_args={"complete_input_dict": data},
        )
        return completion_response

    def convert_to_model_response_object(  # noqa: PLR0915
        self,
        completion_response: Union[List[Dict[str, Any]], Dict[str, Any]],
        model_response: ModelResponse,
        task: Optional[hf_tasks],
        optional_params: dict,
        encoding: Any,
        messages: List[AllMessageValues],
        model: str,
    ):
        if task is None:
            task = "text-generation-inference"  # default to tgi

        if task == "conversational":
            if len(completion_response["generated_text"]) > 0:  # type: ignore
                model_response.choices[0].message.content = completion_response[  # type: ignore
                    "generated_text"
                ]
        elif task == "text-generation-inference":
            if (
                not isinstance(completion_response, list)
                or not isinstance(completion_response[0], dict)
                or "generated_text" not in completion_response[0]
            ):
                raise HuggingfaceError(
                    status_code=422,
                    message=f"response is not in expected format - {completion_response}",
                    headers=None,
                )

            if len(completion_response[0]["generated_text"]) > 0:
                model_response.choices[0].message.content = output_parser(  # type: ignore
                    completion_response[0]["generated_text"]
                )
            ## GETTING LOGPROBS + FINISH REASON
            if (
                "details" in completion_response[0]
                and "tokens" in completion_response[0]["details"]
            ):
                model_response.choices[0].finish_reason = completion_response[0][
                    "details"
                ]["finish_reason"]
                sum_logprob = 0
                for token in completion_response[0]["details"]["tokens"]:
                    if token["logprob"] is not None:
                        sum_logprob += token["logprob"]
                setattr(model_response.choices[0].message, "_logprob", sum_logprob)  # type: ignore
            if "best_of" in optional_params and optional_params["best_of"] > 1:
                if (
                    "details" in completion_response[0]
                    and "best_of_sequences" in completion_response[0]["details"]
                ):
                    choices_list = []
                    for idx, item in enumerate(
                        completion_response[0]["details"]["best_of_sequences"]
                    ):
                        sum_logprob = 0
                        for token in item["tokens"]:
                            if token["logprob"] is not None:
                                sum_logprob += token["logprob"]
                        if len(item["generated_text"]) > 0:
                            message_obj = Message(
                                content=output_parser(item["generated_text"]),
                                logprobs=sum_logprob,
                            )
                        else:
                            message_obj = Message(content=None)
                        choice_obj = Choices(
                            finish_reason=item["finish_reason"],
                            index=idx + 1,
                            message=message_obj,
                        )
                        choices_list.append(choice_obj)
                    model_response.choices.extend(choices_list)
        elif task == "text-classification":
            model_response.choices[0].message.content = json.dumps(  # type: ignore
                completion_response
            )
        else:
            if (
                isinstance(completion_response, list)
                and len(completion_response[0]["generated_text"]) > 0
            ):
                model_response.choices[0].message.content = output_parser(  # type: ignore
                    completion_response[0]["generated_text"]
                )
        ## CALCULATING USAGE
        prompt_tokens = 0
        try:
            prompt_tokens = token_counter(model=model, messages=messages)
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
                )  ##[TODO] use the llama2 tokenizer here
            except Exception:
                # this should remain non blocking we should not block a response returning if calculating usage fails
                pass
        else:
            completion_tokens = 0

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        model_response._hidden_params["original_response"] = completion_response
        return model_response

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        ## Some servers might return streaming responses even though stream was not set to true. (e.g. Baseten)
        task = litellm_params.get("task", None)
        is_streamed = False
        if (
            raw_response.__dict__["headers"].get("Content-Type", "")
            == "text/event-stream"
        ):
            is_streamed = True

        # iterate over the complete streamed response, and return the final answer
        if is_streamed:
            completion_response = self._convert_streamed_response_to_complete_response(
                response=raw_response,
                logging_obj=logging_obj,
                model=model,
                data=request_data,
                api_key=api_key,
            )
        else:
            ## LOGGING
            logging_obj.post_call(
                input=request_data,
                api_key=api_key,
                original_response=raw_response.text,
                additional_args={"complete_input_dict": request_data},
            )
            ## RESPONSE OBJECT
            try:
                completion_response = raw_response.json()
                if isinstance(completion_response, dict):
                    completion_response = [completion_response]
            except Exception:
                raise HuggingfaceError(
                    message=f"Original Response received: {raw_response.text}",
                    status_code=raw_response.status_code,
                )

        if isinstance(completion_response, dict) and "error" in completion_response:
            raise HuggingfaceError(
                message=completion_response["error"],  # type: ignore
                status_code=raw_response.status_code,
            )
        return self.convert_to_model_response_object(
            completion_response=completion_response,
            model_response=model_response,
            task=task if task is not None and task in hf_task_list else None,
            optional_params=optional_params,
            encoding=encoding,
            messages=messages,
            model=model,
        )
