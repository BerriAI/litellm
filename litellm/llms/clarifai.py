import os, types, traceback
import json
import requests
import time
from typing import Callable, Optional
from litellm.utils import ModelResponse, Usage, Choices, Message
import litellm
import httpx
from .prompt_templates.factory import prompt_factory, custom_prompt


class ClarifaiError(Exception):
    def __init__(self, status_code, message, url):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url=url
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )

class ClarifaiConfig:
    """
    Reference: https://clarifai.com/meta/Llama-2/models/llama2-70b-chat
    TODO fill in the details
    """
    max_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_k: Optional[int] = None

    def __init__(
            self,
            max_tokens: Optional[int] = None,
            temperature: Optional[int] = None,
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

def validate_environment(api_key):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

def completions_to_model(payload):
        # if payload["n"] != 1:
        #     raise HTTPException(
        #         status_code=422,
        #         detail="Only one generation is supported. Please set candidate_count to 1.",
        #     )

        params = {}
        if temperature := payload.get("temperature"):
            params["temperature"] = temperature
        if max_tokens := payload.get("max_tokens"):
            params["max_tokens"] = max_tokens
        return {
            "inputs": [{"data": {"text": {"raw": payload["prompt"]}}}],
            "model": {"output_info": {"params": params}},
}

def convert_model_to_url(model: str, api_base: str):
    user_id, app_id, model_id = model.split(".")
    return f"{api_base}/users/{user_id}/apps/{app_id}/models/{model_id}/outputs"

def get_prompt_model_name(url: str):
    clarifai_model_name = url.split("/")[-2]
    if "claude" in clarifai_model_name:
        return "anthropic", clarifai_model_name.replace("_", ".")
    if ("llama" in clarifai_model_name)or ("mistral" in clarifai_model_name):
        return "", "meta-llama/llama-2-chat"
    else:
        return "", clarifai_model_name

def completion(
    model: str,
    messages: list,
    api_base: str,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    custom_prompt_dict={},
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key)
    model = convert_model_to_url(model, api_base)
    prompt = " ".join(message["content"] for message in messages) # TODO

    ## Load Config
    config = litellm.ClarifaiConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):
            optional_params[k] = v

    custom_llm_provider, orig_model_name = get_prompt_model_name(model)
    if custom_llm_provider == "anthropic":
        prompt = prompt_factory(
            model=orig_model_name,
            messages=messages,
            api_key=api_key,
            custom_llm_provider="clarifai"
        )
    else:
        prompt = prompt_factory(
            model=orig_model_name,
            messages=messages,
            api_key=api_key,
            custom_llm_provider=custom_llm_provider
        )
    # print(prompt); exit(0)

    data = {
        "prompt": prompt,
        **optional_params,
    }
    data = completions_to_model(data)


    ## LOGGING
    logging_obj.pre_call(
        input=prompt,
        api_key=api_key,
        additional_args={
            "complete_input_dict": data,
            "headers": headers,
            "api_base": api_base,
        },
    )
    
    ## COMPLETION CALL
    response = requests.post(
        model,
        headers=headers,
        data=json.dumps(data),
    )
    # print(response.content); exit()
    """
    {"status":{"code":10000,"description":"Ok","req_id":"d914cf7e097487997910650cde954a37"},"outputs":[{"id":"c2baa668174b4547bd4d2e9f8996198d","status":{"code":10000,"description":"Ok"},"created_at":"2024-02-07T10:57:52.917990493Z","model":{"id":"GPT-4","name":"GPT-4","created_at":"2023-06-08T17:40:07.964967Z","modified_at":"2023-12-04T11:39:54.587604Z","app_id":"chat-completion","model_version":{"id":"5d7a50b44aec4a01a9c492c5a5fcf387","created_at":"2023-11-09T19:57:56.961259Z","status":{"code":21100,"description":"Model is trained and ready"},"completed_at":"2023-11-09T20:00:48.933172Z","visibility":{"gettable":50},"app_id":"chat-completion","user_id":"openai","metadata":{}},"user_id":"openai","model_type_id":"text-to-text","visibility":{"gettable":50},"toolkits":[],"use_cases":[],"languages":[],"languages_full":[],"check_consents":[],"workflow_recommended":false,"image":{"url":"https://data.clarifai.com/small/users/openai/apps/chat-completion/inputs/image/34326a9914d361bb93ae8e5381689755","hosted":{"prefix":"https://data.clarifai.com","suffix":"users/openai/apps/chat-completion/inputs/image/34326a9914d361bb93ae8e5381689755","sizes":["small"],"crossorigin":"use-credentials"}}},"input":{"id":"fba1f22a332743f083ddae0a7eb443ae","data":{"text":{"raw":"what\'s the weather in SF","url":"https://samples.clarifai.com/placeholder.gif"}}},"data":{"text":{"raw":"As an AI, I\'m unable to provide real-time information or updates. Please check a reliable weather website or app for the current weather in San Francisco.","text_info":{"encoding":"UnknownTextEnc"}}}}]}
    """
    if response.status_code != 200:
        raise ClarifaiError(status_code=response.status_code, message=response.text, url=model)
    if "stream" in optional_params and optional_params["stream"] == True:
        return response.iter_lines()
    else:
        logging_obj.post_call(
            input=prompt,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        ## RESPONSE OBJECT
        completion_response = response.json()
        # print(completion_response)
        try:
            choices_list = []
            for idx, item in enumerate(completion_response["outputs"]):
                if len(item["data"]["text"]["raw"]) > 0:
                    message_obj = Message(content=item["data"]["text"]["raw"])
                else:
                    message_obj = Message(content=None)
                choice_obj = Choices(
                    finish_reason="stop",
                    index=idx + 1, #check
                    message=message_obj,
                )
                choices_list.append(choice_obj)
            model_response["choices"] = choices_list
        except Exception as e:
            raise ClarifaiError(
                message=traceback.format_exc(), status_code=response.status_code, url=model
            )

        # Calculate Usage
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content"))
        )
        model_response["model"] = model
        model_response["usage"] = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return model_response