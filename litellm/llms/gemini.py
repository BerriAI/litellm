import os, types, traceback, copy
import json
from enum import Enum
import time
from typing import Callable, Optional
from litellm.utils import ModelResponse, get_secret, Choices, Message, Usage
import litellm
import sys, httpx
from .prompt_templates.factory import prompt_factory, custom_prompt


class GeminiError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST",
            url="https://developers.generativeai.google/api/python/google/generativeai/chat",
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class GeminiConfig:
    """
    Reference: https://ai.google.dev/api/python/google/generativeai/GenerationConfig

    The class `GeminiConfig` provides configuration for the Gemini's API interface. Here are the parameters:

    - `candidate_count` (int): Number of generated responses to return.

    - `stop_sequences` (List[str]): The set of character sequences (up to 5) that will stop output generation. If specified, the API will stop at the first appearance of a stop sequence. The stop sequence will not be included as part of the response.

    - `max_output_tokens` (int): The maximum number of tokens to include in a candidate. If unset, this will default to output_token_limit specified in the model's specification.

    - `temperature` (float): Controls the randomness of the output. Note: The default value varies by model, see the Model.temperature attribute of the Model returned the genai.get_model function. Values can range from [0.0,1.0], inclusive. A value closer to 1.0 will produce responses that are more varied and creative, while a value closer to 0.0 will typically result in more straightforward responses from the model.

    - `top_p` (float): Optional. The maximum cumulative probability of tokens to consider when sampling.

    - `top_k` (int): Optional. The maximum number of tokens to consider when sampling.
    """

    candidate_count: Optional[int] = None
    stop_sequences: Optional[list] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    def __init__(
        self,
        candidate_count: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
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


def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    api_key,
    encoding,
    logging_obj,
    custom_prompt_dict: dict,
    acompletion: bool = False,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    try:
        import google.generativeai as genai
    except:
        raise Exception(
            "Importing google.generativeai failed, please run 'pip install -q google-generativeai"
        )
    genai.configure(api_key=api_key)

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
        prompt = prompt_factory(
            model=model, messages=messages, custom_llm_provider="gemini"
        )

    ## Load Config
    inference_params = copy.deepcopy(optional_params)
    stream = inference_params.pop("stream", None)
    safety_settings = optional_params.pop("safety_settings", None)
    config = litellm.GeminiConfig.get_config()
    for k, v in config.items():
        if (
            k not in inference_params
        ):  # completion(top_k=3) > gemini_config(top_k=3) <- allows for dynamic variables to be passed in
            inference_params[k] = v

    ## LOGGING
    logging_obj.pre_call(
        input=prompt,
        api_key="",
        additional_args={"complete_input_dict": {"inference_params": inference_params}},
    )
    ## COMPLETION CALL
    try:
        _model = genai.GenerativeModel(f"models/{model}")
        if stream != True:
            response = _model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(**inference_params),
                safety_settings=safety_settings,
            )
        else:
            response = _model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(**inference_params),
                safety_settings=safety_settings,
                stream=True,
            )
            return response
    except Exception as e:
        raise GeminiError(
            message=str(e),
            status_code=500,
        )

    ## LOGGING
    logging_obj.post_call(
        input=prompt,
        api_key="",
        original_response=response,
        additional_args={"complete_input_dict": {}},
    )
    print_verbose(f"raw model_response: {response}")
    ## RESPONSE OBJECT
    completion_response = response
    try:
        choices_list = []
        for idx, item in enumerate(completion_response.candidates):
            if len(item.content.parts) > 0:
                message_obj = Message(content=item.content.parts[0].text)
            else:
                message_obj = Message(content=None)
            choice_obj = Choices(index=idx + 1, message=message_obj)
            choices_list.append(choice_obj)
        model_response["choices"] = choices_list
    except Exception as e:
        traceback.print_exc()
        raise GeminiError(
            message=traceback.format_exc(), status_code=response.status_code
        )

    try:
        completion_response = model_response["choices"][0]["message"].get("content")
        if completion_response is None:
            raise Exception
    except:
        original_response = f"response: {response}"
        if hasattr(response, "candidates"):
            original_response = f"response: {response.candidates}"
            if "SAFETY" in original_response:
                original_response += (
                    "\nThe candidate content was flagged for safety reasons."
                )
            elif "RECITATION" in original_response:
                original_response += (
                    "\nThe candidate content was flagged for recitation reasons."
                )
        raise GeminiError(
            status_code=400,
            message=f"No response received. Original response - {original_response}",
        )

    ## CALCULATING USAGE
    prompt_str = ""
    for m in messages:
        if isinstance(m["content"], str):
            prompt_str += m["content"]
        elif isinstance(m["content"], list):
            for content in m["content"]:
                if content["type"] == "text":
                    prompt_str += content["text"]
    prompt_tokens = len(encoding.encode(prompt_str))
    completion_tokens = len(
        encoding.encode(model_response["choices"][0]["message"].get("content", ""))
    )

    model_response["created"] = int(time.time())
    model_response["model"] = "gemini/" + model
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    model_response.usage = usage
    return model_response


def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
