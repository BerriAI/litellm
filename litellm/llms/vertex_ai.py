import os, types
import json
from enum import Enum
import requests
import time
from typing import Callable, Optional
from litellm.utils import ModelResponse, Usage
import litellm
import httpx

class VertexAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url=" https://cloud.google.com/vertex-ai/")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class VertexAIConfig():
    """
    Reference: https://cloud.google.com/vertex-ai/docs/generative-ai/chat/test-chat-prompts

    The class `VertexAIConfig` provides configuration for the VertexAI's API interface. Below are the parameters:

    - `temperature` (float): This controls the degree of randomness in token selection.

    - `max_output_tokens` (integer): This sets the limitation for the maximum amount of token in the text output. In this case, the default value is 256.

    - `top_p` (float): The tokens are selected from the most probable to the least probable until the sum of their probabilities equals the `top_p` value. Default is 0.95.

    - `top_k` (integer): The value of `top_k` determines how many of the most probable tokens are considered in the selection. For example, a `top_k` of 1 means the selected token is the most probable among all tokens. The default value is 40.

    Note: Please make sure to modify the default parameters as required for your use case.
    """
    temperature: Optional[float]=None
    max_output_tokens: Optional[int]=None
    top_p: Optional[float]=None
    top_k: Optional[int]=None

    def __init__(self, 
                 temperature: Optional[float]=None,
                 max_output_tokens: Optional[int]=None,
                 top_p: Optional[float]=None,
                 top_k: Optional[int]=None) -> None:
        
        locals_ = locals()
        for key, value in locals_.items():
            if key != 'self' and value is not None:
                setattr(self.__class__, key, value)
    
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)) 
                and v is not None}

def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    logging_obj,
    vertex_project=None,
    vertex_location=None,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    try:
        import vertexai
    except:
        raise Exception("vertexai import failed please run `pip install google-cloud-aiplatform`")
    from vertexai.preview.language_models import ChatModel, CodeChatModel, InputOutputTextPair
    from vertexai.language_models import TextGenerationModel, CodeGenerationModel

    vertexai.init(
        project=vertex_project, location=vertex_location
    )

    ## Load Config
    config = litellm.VertexAIConfig.get_config()
    for k, v in config.items(): 
        if k not in optional_params: 
            optional_params[k] = v

    # vertexai does not use an API key, it looks for credentials.json in the environment

    prompt = " ".join([message["content"] for message in messages])

    mode = "" 
    if model in litellm.vertex_chat_models:
        chat_model = ChatModel.from_pretrained(model)
        mode = "chat"
    elif model in litellm.vertex_text_models:
        text_model = TextGenerationModel.from_pretrained(model)
        mode = "text"
    elif model in litellm.vertex_code_text_models:
        text_model = CodeGenerationModel.from_pretrained(model)
        mode = "text"
    else: # vertex_code_chat_models
        chat_model = CodeChatModel.from_pretrained(model)
        mode = "chat"
    
    if mode == "chat":
        chat = chat_model.start_chat()

        ## LOGGING
        logging_obj.pre_call(input=prompt, api_key=None, additional_args={"complete_input_dict": optional_params})

        if "stream" in optional_params and optional_params["stream"] == True:
            # NOTE: VertexAI does not accept stream=True as a param and raises an error,
            # we handle this by removing 'stream' from optional params and sending the request
            # after we get the response we add optional_params["stream"] = True, since main.py needs to know it's a streaming response to then transform it for the OpenAI format
            optional_params.pop("stream", None) # vertex ai raises an error when passing stream in optional params
            model_response = chat.send_message_streaming(prompt, **optional_params)
            optional_params["stream"] = True
            return model_response

        completion_response = chat.send_message(prompt, **optional_params).text
    elif mode == "text":
        ## LOGGING
        logging_obj.pre_call(input=prompt, api_key=None)

        if "stream" in optional_params and optional_params["stream"] == True:
            optional_params.pop("stream", None) # See note above on handling streaming for vertex ai 
            model_response = text_model.predict_streaming(prompt, **optional_params)
            optional_params["stream"] = True
            return model_response

        completion_response = text_model.predict(prompt, **optional_params).text
        
    ## LOGGING
    logging_obj.post_call(
        input=prompt, api_key=None, original_response=completion_response
    )

    ## RESPONSE OBJECT
    if len(str(completion_response)) > 0: 
        model_response["choices"][0]["message"][
            "content"
        ] = str(completion_response)
    model_response["choices"][0]["message"]["content"] = str(completion_response)
    model_response["created"] = time.time()
    model_response["model"] = model
    ## CALCULATING USAGE
    prompt_tokens = len(
        encoding.encode(prompt)
    ) 
    completion_tokens = len(
        encoding.encode(model_response["choices"][0]["message"].get("content", ""))
    )
    usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    model_response.usage = usage
    return model_response


def embedding():
    # logic for parsing in - calling - parsing out model embedding calls
    pass
