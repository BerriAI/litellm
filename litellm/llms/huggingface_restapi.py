## Uses the huggingface text generation inference API
import copy
import enum
import json
import os
import time
import types
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, get_args

import httpx
import requests

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.secret_managers.main import get_secret_str
from litellm.types.completion import ChatCompletionMessageToolCallParam
from litellm.utils import Choices, CustomStreamWrapper, Message, ModelResponse, Usage

from .base import BaseLLM
from .prompt_templates.factory import custom_prompt, prompt_factory


class HuggingfaceError(Exception):
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
                method="POST", url="https://api-inference.huggingface.co/models"
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


hf_task_list = [
    "text-generation-inference",
    "conversational",
    "text-classification",
    "text-generation",
]

hf_tasks = Literal[
    "text-generation-inference",
    "conversational",
    "text-classification",
    "text-generation",
]

hf_tasks_embeddings = Literal[  # pipeline tags + hf tei endpoints - https://huggingface.github.io/text-embeddings-inference/#/
    "sentence-similarity", "feature-extraction", "rerank", "embed", "similarity"
]


class HuggingfaceConfig:
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

    def get_special_options_params(self):
        return ["use_cache", "wait_for_model"]

    def get_supported_openai_params(self):
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
        self, non_default_params: dict, optional_params: dict
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
        return optional_params

    def get_hf_api_key(self) -> Optional[str]:
        return get_secret_str("HUGGINGFACE_API_KEY")


def output_parser(generated_text: str):
    """
    Parse the output text to remove any special characters. In our current approach we just check for ChatML tokens.

    Initial issue that prompted this - https://github.com/BerriAI/litellm/issues/763
    """
    chat_template_tokens = ["<|assistant|>", "<|system|>", "<|user|>", "<s>", "</s>"]
    for token in chat_template_tokens:
        if generated_text.strip().startswith(token):
            generated_text = generated_text.replace(token, "", 1)
        if generated_text.endswith(token):
            generated_text = generated_text[::-1].replace(token[::-1], "", 1)[::-1]
    return generated_text


tgi_models_cache = None
conv_models_cache = None


def read_tgi_conv_models():
    try:
        global tgi_models_cache, conv_models_cache
        # Check if the cache is already populated
        # so we don't keep on reading txt file if there are 1k requests
        if (tgi_models_cache is not None) and (conv_models_cache is not None):
            return tgi_models_cache, conv_models_cache
        # If not, read the file and populate the cache
        tgi_models = set()
        script_directory = os.path.dirname(os.path.abspath(__file__))
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


def get_hf_task_for_model(model: str) -> Tuple[hf_tasks, str]:
    # read text file, cast it to set
    # read the file called "huggingface_llms_metadata/hf_text_generation_models.txt"
    if model.split("/")[0] in hf_task_list:
        split_model = model.split("/", 1)
        return split_model[0], split_model[1]  # type: ignore
    tgi_models, conversational_models = read_tgi_conv_models()
    if model in tgi_models:
        return "text-generation-inference", model
    elif model in conversational_models:
        return "conversational", model
    elif "roneneldan/TinyStories" in model:
        return "text-generation", model
    else:
        return "text-generation-inference", model  # default to tgi


from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


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
    http_client = AsyncHTTPHandler(concurrent_limit=1)

    model_info = await http_client.get(url=api_base)

    model_info_dict = model_info.json()

    pipeline_tag: Optional[str] = model_info_dict.get("pipeline_tag", None)

    return pipeline_tag


class Huggingface(BaseLLM):
    _client_session: Optional[httpx.Client] = None
    _aclient_session: Optional[httpx.AsyncClient] = None

    def __init__(self) -> None:
        super().__init__()

    def _validate_environment(self, api_key, headers) -> dict:
        default_headers = {
            "content-type": "application/json",
        }
        if api_key and headers is None:
            default_headers["Authorization"] = (
                f"Bearer {api_key}"  # Huggingface Inference Endpoint default is to accept bearer tokens
            )
            headers = default_headers
        elif headers:
            headers = headers
        else:
            headers = default_headers
        return headers

    def convert_to_model_response_object(
        self,
        completion_response,
        model_response: litellm.ModelResponse,
        task: hf_tasks,
        optional_params,
        encoding,
        input_text,
        model,
    ):
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
            if len(completion_response[0]["generated_text"]) > 0:
                model_response.choices[0].message.content = output_parser(  # type: ignore
                    completion_response[0]["generated_text"]
                )
        ## CALCULATING USAGE
        prompt_tokens = 0
        try:
            prompt_tokens = len(
                encoding.encode(input_text)
            )  ##[TODO] use the llama2 tokenizer here
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

    def completion(
        self,
        model: str,
        messages: list,
        api_base: Optional[str],
        headers: Optional[dict],
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: float,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        custom_prompt_dict={},
        acompletion: bool = False,
        litellm_params=None,
        logger_fn=None,
    ):
        super().completion()
        exception_mapping_worked = False
        try:
            headers = self._validate_environment(api_key, headers)
            task, model = get_hf_task_for_model(model)
            ## VALIDATE API FORMAT
            if task is None or not isinstance(task, str) or task not in hf_task_list:
                raise Exception(
                    "Invalid hf task - {}. Valid formats - {}.".format(task, hf_tasks)
                )

            print_verbose(f"{model}, {task}")
            completion_url = ""
            input_text = ""
            if "https" in model:
                completion_url = model
            elif api_base:
                completion_url = api_base
            elif "HF_API_BASE" in os.environ:
                completion_url = os.getenv("HF_API_BASE", "")
            elif "HUGGINGFACE_API_BASE" in os.environ:
                completion_url = os.getenv("HUGGINGFACE_API_BASE", "")
            else:
                completion_url = f"https://api-inference.huggingface.co/models/{model}"

            ## Load Config
            config = litellm.HuggingfaceConfig.get_config()
            for k, v in config.items():
                if (
                    k not in optional_params
                ):  # completion(top_k=3) > huggingfaceConfig(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ### MAP INPUT PARAMS
            #### HANDLE SPECIAL PARAMS
            special_params = HuggingfaceConfig().get_special_options_params()
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
                inference_params = copy.deepcopy(optional_params)
                inference_params.pop("details")
                inference_params.pop("return_full_text")
                past_user_inputs = []
                generated_responses = []
                text = ""
                for message in messages:
                    if message["role"] == "user":
                        if text != "":
                            past_user_inputs.append(text)
                        text = message["content"]
                    elif message["role"] == "assistant" or message["role"] == "system":
                        generated_responses.append(message["content"])
                data = {
                    "inputs": {
                        "text": text,
                        "past_user_inputs": past_user_inputs,
                        "generated_responses": generated_responses,
                    },
                    "parameters": inference_params,
                }
                input_text = "".join(message["content"] for message in messages)
            elif task == "text-generation-inference":
                # always send "details" and "return_full_text" as params
                if model in custom_prompt_dict:
                    # check if the model has a registered custom prompt
                    model_prompt_details = custom_prompt_dict[model]
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
                    "inputs": prompt,
                    "parameters": optional_params,
                    "stream": (  # type: ignore
                        True
                        if "stream" in optional_params
                        and isinstance(optional_params["stream"], bool)
                        and optional_params["stream"] is True  # type: ignore
                        else False
                    ),
                }
                input_text = prompt
            else:
                # Non TGI and Conversational llms
                # We need this branch, it removes 'details' and 'return_full_text' from params
                if model in custom_prompt_dict:
                    # check if the model has a registered custom prompt
                    model_prompt_details = custom_prompt_dict[model]
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
                inference_params = copy.deepcopy(optional_params)
                inference_params.pop("details")
                inference_params.pop("return_full_text")
                data = {
                    "inputs": prompt,
                }
                if task == "text-generation-inference":
                    data["parameters"] = inference_params
                    data["stream"] = (  # type: ignore
                        True  # type: ignore
                        if "stream" in optional_params
                        and optional_params["stream"] is True
                        else False
                    )
                input_text = prompt

            ### RE-ADD SPECIAL PARAMS
            if len(special_params_dict.keys()) > 0:
                data.update({"options": special_params_dict})

            ## LOGGING
            logging_obj.pre_call(
                input=input_text,
                api_key=api_key,
                additional_args={
                    "complete_input_dict": data,
                    "task": task,
                    "headers": headers,
                    "api_base": completion_url,
                    "acompletion": acompletion,
                },
            )
            ## COMPLETION CALL

            # SSL certificates (a.k.a CA bundle) used to verify the identity of requested hosts.
            ssl_verify = os.getenv("SSL_VERIFY", litellm.ssl_verify)
            if ssl_verify in ["True", "False"]:
                ssl_verify = bool(ssl_verify)

            if acompletion is True:
                ### ASYNC STREAMING
                if optional_params.get("stream", False):
                    return self.async_streaming(logging_obj=logging_obj, api_base=completion_url, data=data, headers=headers, model_response=model_response, model=model, timeout=timeout)  # type: ignore
                else:
                    ### ASYNC COMPLETION
                    return self.acompletion(api_base=completion_url, data=data, headers=headers, model_response=model_response, task=task, encoding=encoding, input_text=input_text, model=model, optional_params=optional_params, timeout=timeout)  # type: ignore
            ### SYNC STREAMING
            if "stream" in optional_params and optional_params["stream"] is True:
                response = requests.post(
                    completion_url,
                    headers=headers,
                    data=json.dumps(data),
                    stream=optional_params["stream"],
                    verify=ssl_verify,
                )
                return response.iter_lines()
            ### SYNC COMPLETION
            else:
                response = requests.post(
                    completion_url,
                    headers=headers,
                    data=json.dumps(data),
                    verify=ssl_verify,
                )

                ## Some servers might return streaming responses even though stream was not set to true. (e.g. Baseten)
                is_streamed = False
                if (
                    response.__dict__["headers"].get("Content-Type", "")
                    == "text/event-stream"
                ):
                    is_streamed = True

                # iterate over the complete streamed response, and return the final answer
                if is_streamed:
                    streamed_response = CustomStreamWrapper(
                        completion_stream=response.iter_lines(),
                        model=model,
                        custom_llm_provider="huggingface",
                        logging_obj=logging_obj,
                    )
                    content = ""
                    for chunk in streamed_response:
                        content += chunk["choices"][0]["delta"]["content"]
                    completion_response: List[Dict[str, Any]] = [
                        {"generated_text": content}
                    ]
                    ## LOGGING
                    logging_obj.post_call(
                        input=input_text,
                        api_key=api_key,
                        original_response=completion_response,
                        additional_args={"complete_input_dict": data, "task": task},
                    )
                else:
                    ## LOGGING
                    logging_obj.post_call(
                        input=input_text,
                        api_key=api_key,
                        original_response=response.text,
                        additional_args={"complete_input_dict": data, "task": task},
                    )
                    ## RESPONSE OBJECT
                    try:
                        completion_response = response.json()
                        if isinstance(completion_response, dict):
                            completion_response = [completion_response]
                    except Exception:
                        import traceback

                        raise HuggingfaceError(
                            message=f"Original Response received: {response.text}; Stacktrace: {traceback.format_exc()}",
                            status_code=response.status_code,
                        )
                print_verbose(f"response: {completion_response}")
                if (
                    isinstance(completion_response, dict)
                    and "error" in completion_response
                ):
                    print_verbose(f"completion error: {completion_response['error']}")  # type: ignore
                    print_verbose(f"response.status_code: {response.status_code}")
                    raise HuggingfaceError(
                        message=completion_response["error"],  # type: ignore
                        status_code=response.status_code,
                    )
                return self.convert_to_model_response_object(
                    completion_response=completion_response,
                    model_response=model_response,
                    task=task,
                    optional_params=optional_params,
                    encoding=encoding,
                    input_text=input_text,
                    model=model,
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
        task: hf_tasks,
        encoding: Any,
        input_text: str,
        model: str,
        optional_params: dict,
        timeout: float,
    ):
        # SSL certificates (a.k.a CA bundle) used to verify the identity of requested hosts.
        ssl_verify = os.getenv("SSL_VERIFY", litellm.ssl_verify)

        response = None
        try:
            async with httpx.AsyncClient(timeout=timeout, verify=ssl_verify) as client:
                response = await client.post(url=api_base, json=data, headers=headers)
                response_json = response.json()
                if response.status_code != 200:
                    if "error" in response_json:
                        raise HuggingfaceError(
                            status_code=response.status_code,
                            message=response_json["error"],
                            request=response.request,
                            response=response,
                        )
                    else:
                        raise HuggingfaceError(
                            status_code=response.status_code,
                            message=response.text,
                            request=response.request,
                            response=response,
                        )

                ## RESPONSE OBJECT
                return self.convert_to_model_response_object(
                    completion_response=response_json,
                    model_response=model_response,
                    task=task,
                    encoding=encoding,
                    input_text=input_text,
                    model=model,
                    optional_params=optional_params,
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
        model: str,
        timeout: float,
    ):
        # SSL certificates (a.k.a CA bundle) used to verify the identity of requested hosts.
        ssl_verify = os.getenv("SSL_VERIFY", litellm.ssl_verify)

        async with httpx.AsyncClient(timeout=timeout, verify=ssl_verify) as client:
            response = client.stream(
                "POST", url=f"{api_base}", json=data, headers=headers
            )
            async with response as r:
                if r.status_code != 200:
                    text = await r.aread()
                    raise HuggingfaceError(
                        status_code=r.status_code,
                        message=str(text),
                    )
                """
                Check first chunk for error message. 
                If error message, raise error. 
                If not - add back to stream
                """
                # Async iterator over the lines in the response body
                response_iterator = r.aiter_lines()

                # Attempt to get the first line/chunk from the response
                try:
                    first_chunk = await response_iterator.__anext__()
                except StopAsyncIteration:
                    # Handle the case where there are no lines to read (empty response)
                    first_chunk = ""

                # Check the first chunk for an error message
                if (
                    "error" in first_chunk.lower()
                ):  # Adjust this condition based on how error messages are structured
                    raise HuggingfaceError(
                        status_code=400,
                        message=first_chunk,
                    )

                # Create a new async generator that begins with the first_chunk and includes the remaining items
                async def custom_stream_with_first_chunk():
                    yield first_chunk  # Yield back the first chunk
                    async for (
                        chunk
                    ) in response_iterator:  # Continue yielding the rest of the chunks
                        yield chunk

                # Creating a new completion stream that starts with the first chunk
                completion_stream = custom_stream_with_first_chunk()

                streamwrapper = CustomStreamWrapper(
                    completion_stream=completion_stream,
                    model=model,
                    custom_llm_provider="huggingface",
                    logging_obj=logging_obj,
                )

                async for transformed_chunk in streamwrapper:
                    yield transformed_chunk

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
        model_response: litellm.EmbeddingResponse,
        model: str,
        input: List,
        encoding: Any,
    ) -> litellm.EmbeddingResponse:
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
            client = AsyncHTTPHandler(concurrent_limit=1)

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
        model_response: litellm.EmbeddingResponse,
        optional_params: dict,
        logging_obj: LiteLLMLoggingObj,
        encoding: Callable,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Union[float, httpx.Timeout] = httpx.Timeout(None),
        aembedding: Optional[bool] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> litellm.EmbeddingResponse:
        super().embedding()
        headers = self._validate_environment(api_key, headers=None)
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
