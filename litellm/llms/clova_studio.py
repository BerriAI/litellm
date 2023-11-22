import os, types
import json
from enum import Enum
import requests
import time
from typing import Callable, List, Optional
import httpx
from litellm.utils import ModelResponse, Usage
import litellm

class ClovaStudioError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://clovastudio.apigw.ntruss.com")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class ClovaStudioConfig():
    """
    Reference: https://api.ncloud-docs.com/docs/en/ai-naver-clovastudio-summary

    The class `ClovaStudioConfig` provides configuration for the ClovaStudio's API interface. Below are the parameters:

    - `topK` (int): Selects and samples the top k token candidates with the highest probabilities from the generated token pool. 0-128 (default value: 0)

    - `includeAiFilters` (bool): Whether to return AI filters in the response.

    - `maxTokens` (int): Maximum number of tokens generated. 0-4096 (default value: 100)

    - `temperature` (float): Degree of diversity for generated tokens (higher setting values generate more diverse sentences). 0-1 (default value: 0.5)

    - `stopBefore` (List[str]): Character that halts token generation. (default value: [])

    - `repeatPenalty` (int): The degree of penalty applied when generating the same token (a higher setting reduces the likelihood of repeatedly generating the same result). 0-10 (default value: 5)

    - `topP` (float): Samples based on the cumulative probabilities of the generated token candidates. 0 < topP < 1 (default value: 0.8)

    Note: Please make sure to modify the default parameters as required for your use case.
    """
    topK: Optional[int]=None
    includeAiFilters: Optional[bool]=None
    maxTokens: Optional[int]=None
    temperature: Optional[float]=None
    stopBefore: Optional[List[str]]=None
    repeatPenalty: Optional[int]=None
    topP: Optional[float]=None

    def __init__(self, 
                topK: Optional[int]=None,
                includeAiFilters: Optional[bool]=None,
                maxTokens: Optional[int]=None,
                temperature: Optional[float]=None,
                stopBefore: Optional[List[str]]=None,
                repeatPenalty: Optional[int]=None,
                topP: Optional[float]=None) -> None:    
        
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


def validate_environment(clova_studio_api_key: str, apigw_api_key: str):
    if clova_studio_api_key is None or apigw_api_key is None:
        raise ValueError(
            "Missing CLOVA Studio API Key - A request to CLOVA Studio has been initiated, but the required API keys are missing. Please ensure that both `NCP_CLOVA_STUDIO_API_KEY` and `NCP_APIGW_API_KEY` are correctly set as environment variables."
        )
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "X-NCP-CLOVASTUDIO-API-KEY": clova_studio_api_key,
        "X-NCP-APIGW-API-KEY": apigw_api_key,
    }
    return headers

def completion(
    model: str,
    messages: list,
    api_base: str, 
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    clova_studio_api_key,
    apigw_api_key,
    clova_studio_project,
    logging_obj,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
    acompletion: bool = False,
):
    headers = validate_environment(clova_studio_api_key, apigw_api_key)

    if clova_studio_project is None:
        raise ValueError(
            "Missing CLOVA Studio Project - A request to CLOVA Studio has been initiated, but the required project name is missing. Please ensure that `NCP_CLOVA_STUDIO_PROJECT` is correctly set as an environment variable."
        )

    ## Load Config
    config = litellm.ClovaStudioConfig.get_config()
    for k, v in config.items(): 
        if k not in optional_params: 
            optional_params[k] = v

    # Check if optional_params is valid for CLOVA Studio
    for k, v in optional_params.items():
         if k not in ClovaStudioConfig.__dict__:
            raise ValueError(
                f"Invalid parameter: {k} is not a valid parameter for ClovaStudioConfig. Please refer to the documentation for more details. https://api.ncloud-docs.com/docs/en/ai-naver-clovastudio-summary"
            )
        

    data = {
        "messages": messages,
        **optional_params,
    }

    ## LOGGING
    logging_obj.pre_call(
        input=messages,
        api_key=None,
    )

    if acompletion:
        # ASYNC COMPLETION CALL
        return async_completion(
            model=model,
            messages=messages,
            api_base=api_base,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            clova_studio_api_key=clova_studio_api_key,
            apigw_api_key=apigw_api_key,
            clova_studio_project=clova_studio_project,
            logging_obj=logging_obj,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
        )
    
    ## COMPLETION CALL
    response = requests.post(
        f"{api_base}/{clova_studio_project}/v1/chat-completions/{model}",
        headers=headers,
        data=json.dumps(data)
    )
    ## LOGGING
    return log_and_format_model_response(
        model=model,
        messages=messages,
        model_response=model_response,
        print_verbose=print_verbose,
        logging_obj=logging_obj,
        response=response,
    )


async def async_completion(
    model: str,
    messages: list,
    api_base: str, 
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    clova_studio_api_key,
    apigw_api_key,
    clova_studio_project,
    logging_obj,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(clova_studio_api_key, apigw_api_key)

    ## Load Config
    config = litellm.ClovaStudioConfig.get_config()
    for k, v in config.items(): 
        if k not in optional_params: 
            optional_params[k] = v

    data = {
        "messages": messages,
        **optional_params,
    }
    
    ## LOGGING
    logging_obj.pre_call(
        input=messages,
        api_key=None,
    )

    ## COMPLETION CALL
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api_base}/{clova_studio_project}/v1/chat-completions/{model}",
            headers=headers,
            data=json.dumps(data)
        )
    ## LOGGING
    return log_and_format_model_response(
        model=model,
        messages=messages,
        model_response=model_response,
        print_verbose=print_verbose,
        logging_obj=logging_obj,
        response=response,
    )


def log_and_format_model_response(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    logging_obj,
    response,
):
        ## LOGGING
    logging_obj.post_call(
            input=messages,
            api_key=None,
            original_response=response.text,
        )
    print_verbose(f"raw model_response: {response.text}")
    ## RESPONSE OBJECT
    if response.status_code != 200:
        raise ClovaStudioError(
            status_code=response.status_code, message=response.text
        )
    completion_response = response.json()

    if int(completion_response["status"]["code"]) != 20000:
        raise ClovaStudioError(
            message=json.dumps(completion_response),
            status_code=response.status_code,
        )

    model_response["choices"][0]["message"]["content"] = completion_response["result"]["message"]["content"]

    ## CALCULATING USAGE
    prompt_tokens = completion_response["result"]["inputLength"]
    completion_tokens = completion_response["result"]["outputLength"]
    if "stopReason" in completion_response["result"]:
        model_response.choices[0].finish_reason = completion_response["result"]["stopReason"]
    model_response["created"] = time.time()
    model_response["model"] = model
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
