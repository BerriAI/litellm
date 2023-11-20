# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you ! We ❤️ you! - Krrish & Ishaan 

import os, openai, sys, json, inspect, uuid, datetime, threading
from typing import Any
from functools import partial
import dotenv, traceback, random, asyncio, time, contextvars
from copy import deepcopy
import httpx
import litellm
from litellm import (  # type: ignore
    client,
    exception_type,
    get_optional_params,
    get_litellm_params,
    Logging,
)
from litellm.utils import (
    get_secret,
    CustomStreamWrapper,
    read_config_args,
    completion_with_fallbacks,
    get_llm_provider,
    get_api_key,
    mock_completion_streaming_obj
)
from .llms import (
    anthropic,
    together_ai,
    ai21,
    sagemaker,
    bedrock,
    huggingface_restapi,
    replicate,
    aleph_alpha,
    nlp_cloud,
    baseten,
    vllm,
    ollama,
    cohere,
    petals,
    oobabooga,
    palm,
    vertex_ai,
    maritalk)
from .llms.openai import OpenAIChatCompletion, OpenAITextCompletion
from .llms.azure import AzureChatCompletion
from .llms.huggingface_restapi import Huggingface
from .llms.prompt_templates.factory import prompt_factory, custom_prompt, function_call_prompt
import tiktoken
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Dict, Union, Mapping

encoding = tiktoken.get_encoding("cl100k_base")
from litellm.utils import (
    get_secret,
    CustomStreamWrapper,
    TextCompletionStreamWrapper,
    ModelResponse,
    TextCompletionResponse,
    TextChoices,
    EmbeddingResponse,
    read_config_args,
    Choices, 
    Message
)

####### ENVIRONMENT VARIABLES ###################
dotenv.load_dotenv()  # Loading env variables using dotenv
openai_chat_completions = OpenAIChatCompletion()
openai_text_completions = OpenAITextCompletion()
azure_chat_completions = AzureChatCompletion()
huggingface = Huggingface()
####### COMPLETION ENDPOINTS ################

class LiteLLM:

  def __init__(self, *, 
               api_key=None, 
               organization: Optional[str] = None,
               base_url: Optional[str]= None,
               timeout: Optional[float] = 600,
               max_retries: Optional[int] = litellm.num_retries,
               default_headers: Optional[Mapping[str, str]] = None,):
    self.params = locals()
    self.chat = Chat(self.params)

class Chat():

  def __init__(self, params):
    self.params = params
    self.completions = Completions(self.params)

class Completions():
  
  def __init__(self, params):
    self.params = params

  def create(self, model, messages, **kwargs):
    for k, v in kwargs.items():
        self.params[k] = v
    response = completion(model=model, messages=messages, **self.params)
    return response

@client
async def acompletion(*args, **kwargs):
    """
    Asynchronously executes a litellm.completion() call for any of litellm supported llms (example gpt-4, gpt-3.5-turbo, claude-2, command-nightly)

    Parameters:
        model (str): The name of the language model to use for text completion. see all supported LLMs: https://docs.litellm.ai/docs/providers/
        messages (List): A list of message objects representing the conversation context (default is an empty list).

        OPTIONAL PARAMS
        functions (List, optional): A list of functions to apply to the conversation messages (default is an empty list).
        function_call (str, optional): The name of the function to call within the conversation (default is an empty string).
        temperature (float, optional): The temperature parameter for controlling the randomness of the output (default is 1.0).
        top_p (float, optional): The top-p parameter for nucleus sampling (default is 1.0).
        n (int, optional): The number of completions to generate (default is 1).
        stream (bool, optional): If True, return a streaming response (default is False).
        stop(string/list, optional): - Up to 4 sequences where the LLM API will stop generating further tokens.
        max_tokens (integer, optional): The maximum number of tokens in the generated completion (default is infinity).
        presence_penalty (float, optional): It is used to penalize new tokens based on their existence in the text so far.
        frequency_penalty: It is used to penalize new tokens based on their frequency in the text so far.
        logit_bias (dict, optional): Used to modify the probability of specific tokens appearing in the completion.
        user (str, optional):  A unique identifier representing your end-user. This can help the LLM provider to monitor and detect abuse.
        metadata (dict, optional): Pass in additional metadata to tag your completion calls - eg. prompt version, details, etc. 
        api_base (str, optional): Base URL for the API (default is None).
        api_version (str, optional): API version (default is None).
        api_key (str, optional): API key (default is None).
        model_list (list, optional): List of api base, version, keys

        LITELLM Specific Params
        mock_response (str, optional): If provided, return a mock completion response for testing or debugging purposes (default is None).
        force_timeout (int, optional): The maximum execution time in seconds for the completion request (default is 600).
        custom_llm_provider (str, optional): Used for Non-OpenAI LLMs, Example usage for bedrock, set model="amazon.titan-tg1-large" and custom_llm_provider="bedrock"
    Returns:
        ModelResponse: A response object containing the generated completion and associated metadata.

    Notes:
        - This function is an asynchronous version of the `completion` function.
        - The `completion` function is called using `run_in_executor` to execute synchronously in the event loop.
        - If `stream` is True, the function returns an async generator that yields completion lines.
    """
    loop = asyncio.get_event_loop()
    model = args[0] if len(args) > 0 else kwargs["model"]
    ### PASS ARGS TO COMPLETION ### 
    kwargs["acompletion"] = True
    try: 
        # Use a partial function to pass your keyword arguments
        func = partial(completion, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(model=model, api_base=kwargs.get("api_base", None))

        if (custom_llm_provider == "openai" 
            or custom_llm_provider == "azure" 
            or custom_llm_provider == "custom_openai"
            or custom_llm_provider == "text-completion-openai"
            or custom_llm_provider == "huggingface"): # currently implemented aiohttp calls for just azure and openai, soon all. 
            if kwargs.get("stream", False): 
                response = completion(*args, **kwargs)
            else:
                # Await normally
                init_response = completion(*args, **kwargs)
                if isinstance(init_response, dict) or isinstance(init_response, ModelResponse): ## CACHING SCENARIO 
                    response = init_response
                elif asyncio.iscoroutine(init_response):
                    response = await init_response
        else: 
            # Call the synchronous function using run_in_executor
            response =  await loop.run_in_executor(None, func_with_context)
        if kwargs.get("stream", False): # return an async generator
            return _async_streaming(response=response, model=model, custom_llm_provider=custom_llm_provider, args=args)
        else: 
            return response
    except Exception as e: 
        raise exception_type(
                model=model, custom_llm_provider=custom_llm_provider, original_exception=e, completion_kwargs=args,
            )

async def _async_streaming(response, model, custom_llm_provider, args): 
    try: 
        async for line in response: 
            yield line
    except Exception as e: 
        raise exception_type(
                model=model, custom_llm_provider=custom_llm_provider, original_exception=e, completion_kwargs=args,
            )

def mock_completion(model: str, messages: List, stream: Optional[bool] = False, mock_response: str = "This is a mock request", **kwargs):
    """
    Generate a mock completion response for testing or debugging purposes.

    This is a helper function that simulates the response structure of the OpenAI completion API.

    Parameters:
        model (str): The name of the language model for which the mock response is generated.
        messages (List): A list of message objects representing the conversation context.
        stream (bool, optional): If True, returns a mock streaming response (default is False).
        mock_response (str, optional): The content of the mock response (default is "This is a mock request").
        **kwargs: Additional keyword arguments that can be used but are not required.

    Returns:
        litellm.ModelResponse: A ModelResponse simulating a completion response with the specified model, messages, and mock response.

    Raises:
        Exception: If an error occurs during the generation of the mock completion response.

    Note:
        - This function is intended for testing or debugging purposes to generate mock completion responses.
        - If 'stream' is True, it returns a response that mimics the behavior of a streaming completion.
    """
    try:
        model_response = ModelResponse(stream=stream)
        if stream is True:
            # don't try to access stream object,
            response = mock_completion_streaming_obj(model_response, mock_response=mock_response, model=model)
            return response
        
        model_response["choices"][0]["message"]["content"] = mock_response
        model_response["created"] = time.time()
        model_response["model"] = model
        return model_response

    except:
        traceback.print_exc()
        raise Exception("Mock completion response failed")

@client
def completion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: List = [],
    functions: List = [],
    function_call: str = "",  # optional params
    timeout: Optional[Union[float, int]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stop=None,
    max_tokens: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float]=None,
    logit_bias: dict = {},
    user: str = "",
    # openai v1.0+ new params
    response_format: Optional[dict] = None,
    seed: Optional[int] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[str] = None,
    deployment_id = None,
    # set api_base, api_version, api_key
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    model_list: Optional[list] = None, # pass in a list of api_base,keys, etc. 

    # Optional liteLLM function params
    **kwargs,
) -> ModelResponse:
    """
    Perform a completion() using any of litellm supported llms (example gpt-4, gpt-3.5-turbo, claude-2, command-nightly)
    Parameters:
        model (str): The name of the language model to use for text completion. see all supported LLMs: https://docs.litellm.ai/docs/providers/
        messages (List): A list of message objects representing the conversation context (default is an empty list).

        OPTIONAL PARAMS
        functions (List, optional): A list of functions to apply to the conversation messages (default is an empty list).
        function_call (str, optional): The name of the function to call within the conversation (default is an empty string).
        temperature (float, optional): The temperature parameter for controlling the randomness of the output (default is 1.0).
        top_p (float, optional): The top-p parameter for nucleus sampling (default is 1.0).
        n (int, optional): The number of completions to generate (default is 1).
        stream (bool, optional): If True, return a streaming response (default is False).
        stop(string/list, optional): - Up to 4 sequences where the LLM API will stop generating further tokens.
        max_tokens (integer, optional): The maximum number of tokens in the generated completion (default is infinity).
        presence_penalty (float, optional): It is used to penalize new tokens based on their existence in the text so far.
        frequency_penalty: It is used to penalize new tokens based on their frequency in the text so far.
        logit_bias (dict, optional): Used to modify the probability of specific tokens appearing in the completion.
        user (str, optional):  A unique identifier representing your end-user. This can help the LLM provider to monitor and detect abuse.
        metadata (dict, optional): Pass in additional metadata to tag your completion calls - eg. prompt version, details, etc. 
        api_base (str, optional): Base URL for the API (default is None).
        api_version (str, optional): API version (default is None).
        api_key (str, optional): API key (default is None).
        model_list (list, optional): List of api base, version, keys

        LITELLM Specific Params
        mock_response (str, optional): If provided, return a mock completion response for testing or debugging purposes (default is None).
        custom_llm_provider (str, optional): Used for Non-OpenAI LLMs, Example usage for bedrock, set model="amazon.titan-tg1-large" and custom_llm_provider="bedrock"
        max_retries (int, optional): The number of retries to attempt (default is 0).
    Returns:
        ModelResponse: A response object containing the generated completion and associated metadata.

    Note:
        - This function is used to perform completions() using the specified language model.
        - It supports various optional parameters for customizing the completion behavior.
        - If 'mock_response' is provided, a mock completion response is returned for testing or debugging.
    """
    ######### unpacking kwargs #####################
    args = locals()
    api_base = kwargs.get('api_base', None)
    return_async = kwargs.get('return_async', False)
    mock_response = kwargs.get('mock_response', None)
    force_timeout= kwargs.get('force_timeout', 600) ## deprecated
    logger_fn = kwargs.get('logger_fn', None)
    verbose = kwargs.get('verbose', False)
    custom_llm_provider = kwargs.get('custom_llm_provider', None)
    litellm_logging_obj = kwargs.get('litellm_logging_obj', None)
    id = kwargs.get('id', None)
    metadata = kwargs.get('metadata', None)
    fallbacks = kwargs.get('fallbacks', None)
    headers = kwargs.get("headers", None)
    num_retries = kwargs.get("num_retries", None) ## deprecated
    max_retries = kwargs.get("max_retries", None)
    context_window_fallback_dict = kwargs.get("context_window_fallback_dict", None)
    ### CUSTOM PROMPT TEMPLATE ### 
    initial_prompt_value = kwargs.get("initial_prompt_value", None)
    roles = kwargs.get("roles", None)
    final_prompt_value = kwargs.get("final_prompt_value", None)
    bos_token = kwargs.get("bos_token", None)
    eos_token = kwargs.get("eos_token", None)
    acompletion = kwargs.get("acompletion", False)
    ######## end of unpacking kwargs ###########
    openai_params = ["functions", "function_call", "temperature", "temperature", "top_p", "n", "stream", "stop", "max_tokens", "presence_penalty", "frequency_penalty", "logit_bias", "user", "request_timeout", "api_base", "api_version", "api_key", "deployment_id", "organization", "base_url", "default_headers", "timeout", "response_format", "seed", "tools", "tool_choice"]
    litellm_params = ["metadata", "acompletion", "caching", "return_async", "mock_response", "api_key", "api_version", "api_base", "force_timeout", "logger_fn", "verbose", "custom_llm_provider", "litellm_logging_obj", "litellm_call_id", "use_client", "id", "fallbacks", "azure", "headers", "model_list", "num_retries", "context_window_fallback_dict", "roles", "final_prompt_value", "bos_token", "eos_token", "request_timeout", "complete_response", "self", "max_retries"]
    default_params = openai_params + litellm_params
    non_default_params = {k: v for k,v in kwargs.items() if k not in default_params} # model-specific params - pass them straight to the model/provider
    
    if mock_response:
        return mock_completion(model, messages, stream=stream, mock_response=mock_response)
    if timeout is None: 
        timeout = 600 # set timeout for 10 minutes by default 
    timeout = float(timeout)
    try:
        if base_url: 
            api_base = base_url
        if max_retries: 
            num_retries = max_retries
        logging = litellm_logging_obj
        fallbacks = (
            fallbacks
            or litellm.model_fallbacks
        )
        if fallbacks is not None:
            return completion_with_fallbacks(**args)
        if model_list is not None: 
            deployments = [m["litellm_params"] for m in model_list if m["model_name"] == model]
            return batch_completion_models(deployments=deployments, **args)
        if litellm.model_alias_map and model in litellm.model_alias_map:
            args["model_alias_map"] = litellm.model_alias_map
            model = litellm.model_alias_map[
                model
            ]  # update the model to the actual value if an alias has been passed in

        model_response = ModelResponse()

        if kwargs.get('azure', False) == True: # don't remove flag check, to remain backwards compatible for repos like Codium
            custom_llm_provider="azure"
        if deployment_id != None: # azure llms 
                model=deployment_id
                custom_llm_provider="azure"
        model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider, api_base=api_base)
        custom_prompt_dict = {} # type: ignore
        if initial_prompt_value or roles or final_prompt_value or bos_token or eos_token:
            custom_prompt_dict = {model: {}}
            if initial_prompt_value:
                custom_prompt_dict[model]["initial_prompt_value"] = initial_prompt_value
            if roles: 
                custom_prompt_dict[model]["roles"] = roles
            if final_prompt_value: 
                custom_prompt_dict[model]["final_prompt_value"] = final_prompt_value
            if bos_token:
                custom_prompt_dict[model]["bos_token"] = bos_token
            if eos_token:
                custom_prompt_dict[model]["eos_token"] = eos_token
        model_api_key = get_api_key(llm_provider=custom_llm_provider, dynamic_api_key=api_key) # get the api key from the environment if required for the model
        if model_api_key and "sk-litellm" in model_api_key:
            api_base = "https://proxy.litellm.ai"
            custom_llm_provider = "openai" 
            api_key = model_api_key
        
        # check if user passed in any of the OpenAI optional params
        optional_params = get_optional_params(
                functions=functions,
                function_call=function_call,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                user=user,
                # params to identify the model
                model=model,
                custom_llm_provider=custom_llm_provider,
                response_format=response_format,
                seed=seed,
                tools=tools,
                tool_choice=tool_choice,
                **non_default_params 
            )
        
        if litellm.add_function_to_prompt and optional_params.get("functions_unsupported_model", None):  # if user opts to add it to prompt, when API doesn't support function calling 
            functions_unsupported_model = optional_params.pop("functions_unsupported_model")
            messages = function_call_prompt(messages=messages, functions=functions_unsupported_model)

        # For logging - save the values of the litellm-specific params passed in
        litellm_params = get_litellm_params(
            return_async=return_async,
            api_key=api_key,
            force_timeout=force_timeout,
            logger_fn=logger_fn,
            verbose=verbose,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            litellm_call_id=kwargs.get('litellm_call_id', None),
            model_alias_map=litellm.model_alias_map,
            completion_call_id=id,
            metadata=metadata
        )
        logging.update_environment_variables(model=model, user=user, optional_params=optional_params, litellm_params=litellm_params)
        if custom_llm_provider == "azure":
            # azure configs
            api_type = get_secret("AZURE_API_TYPE") or "azure"

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("AZURE_API_BASE")
            )

            api_version = (
                api_version or
                litellm.api_version or
                get_secret("AZURE_API_VERSION")
            )

            api_key = (
                api_key or
                litellm.api_key or
                litellm.azure_key or
                get_secret("AZURE_API_KEY")
            )

            azure_ad_token = (
                optional_params.pop("azure_ad_token", None) or
                get_secret("AZURE_AD_TOKEN")
            )

            headers = (
                headers or
                litellm.headers
            )

            ## LOAD CONFIG - if set
            config=litellm.AzureOpenAIConfig.get_config()
            for k, v in config.items():
                if k not in optional_params: # completion(top_k=3) > azure_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ## COMPLETION CALL
            response = azure_chat_completions.completion(
                model=model,
                messages=messages,
                headers=headers,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                api_type=api_type,
                azure_ad_token=azure_ad_token,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                logging_obj=logging, 
                acompletion=acompletion, 
                timeout=timeout
            )

            ## LOGGING
            logging.post_call(
                input=messages,
                api_key=api_key,
                original_response=response,
                additional_args={
                    "headers": headers,
                    "api_version": api_version,
                    "api_base": api_base,
                },
            )
        elif (
            model in litellm.open_ai_chat_completion_models
            or custom_llm_provider == "custom_openai"
            or custom_llm_provider == "openai"
            or "ft:gpt-3.5-turbo" in model  # finetune gpt-3.5-turbo
        ):  # allow user to make an openai call with a custom base
            # note: if a user sets a custom base - we should ensure this works
            # allow for the setting of dynamic and stateful api-bases
            api_base = (
                api_base
                or litellm.api_base
                or get_secret("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            openai.organization = (
                litellm.organization
                or get_secret("OPENAI_ORGANIZATION")
                or None # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
            )
            # set API KEY
            api_key = (
                api_key or
                dynamic_api_key or # allows us to read env variables for compatible openai api's like perplexity 
                litellm.api_key or
                litellm.openai_key or
                get_secret("OPENAI_API_KEY")
            )

            headers = (
                    headers or
                    litellm.headers
            )

            ## LOAD CONFIG - if set
            config=litellm.OpenAIConfig.get_config()
            for k, v in config.items():
                if k not in optional_params: # completion(top_k=3) > openai_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ## COMPLETION CALL
            try:
                response = openai_chat_completions.completion(
                    model=model,
                    messages=messages,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    api_key=api_key,
                    api_base=api_base,
                    acompletion=acompletion,
                    logging_obj=logging,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    timeout=timeout
                )
            except Exception as e:
                ## LOGGING - log the original exception returned
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=str(e),
                    additional_args={"headers": headers},
                )
                raise e

            ## LOGGING
            logging.post_call(
                input=messages,
                api_key=api_key,
                original_response=response,
                additional_args={"headers": headers},
            )
        elif (
            custom_llm_provider == "text-completion-openai"
            or "ft:babbage-002" in model
            or "ft:davinci-002" in model  # support for finetuned completion models
        ):
            # print("calling custom openai provider")
            openai.api_type = "openai"

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )

            openai.api_version = None
            # set API KEY

            api_key = (
                api_key or
                litellm.api_key or
                litellm.openai_key or
                get_secret("OPENAI_API_KEY")
            )

            headers = (
                headers or
                litellm.headers
            )

            ## LOAD CONFIG - if set
            config=litellm.OpenAITextCompletionConfig.get_config()
            for k, v in config.items():
                if k not in optional_params: # completion(top_k=3) > openai_text_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v
            if litellm.organization:
                openai.organization = litellm.organization

            if len(messages)>0 and "content" in messages[0] and type(messages[0]["content"]) == list: 
                # text-davinci-003 can accept a string or array, if it's an array, assume the array is set in messages[0]['content']
                # https://platform.openai.com/docs/api-reference/completions/create
                prompt = messages[0]["content"]
            else:
                prompt = " ".join([message["content"] for message in messages]) # type: ignore
            ## LOGGING
            logging.pre_call(
                input=prompt,
                api_key=api_key,
                additional_args={
                    "openai_organization": litellm.organization,
                    "headers": headers,
                    "api_base": api_base,
                    "api_type": openai.api_type,
                },
            )
            ## COMPLETION CALL
            model_response = openai_text_completions.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                api_key=api_key,
                api_base=api_base,
                acompletion=acompletion,
                logging_obj=logging,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn
            )
            
            # if "stream" in optional_params and optional_params["stream"] == True:
            #     response = CustomStreamWrapper(model_response, model, custom_llm_provider="text-completion-openai", logging_obj=logging)
            #     return response
            response = model_response
        elif (
            "replicate" in model or 
            custom_llm_provider == "replicate" or
            model in litellm.replicate_models
        ):
            # Setting the relevant API KEY for replicate, replicate defaults to using os.environ.get("REPLICATE_API_TOKEN")
            replicate_key = None
            replicate_key = (
                api_key
                or litellm.replicate_key
                or litellm.api_key 
                or get_secret("REPLICATE_API_KEY")
                or get_secret("REPLICATE_API_TOKEN")
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("REPLICATE_API_BASE")
                or "https://api.replicate.com/v1"
            )

            model_response = replicate.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding, # for calculating input/output tokens
                api_key=replicate_key,
                logging_obj=logging, 
            )
            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(model_response, model, logging_obj=logging, custom_llm_provider="replicate")
                return response
            response = model_response

        elif custom_llm_provider=="anthropic":
            api_key = (
                api_key 
                or litellm.anthropic_key 
                or litellm.api_key
                or os.environ.get("ANTHROPIC_API_KEY") 
            )
            api_base = (
                api_base
                or litellm.api_base
                or get_secret("ANTHROPIC_API_BASE")
                or "https://api.anthropic.com/v1/complete"
            )
            custom_prompt_dict = (
                custom_prompt_dict
                or litellm.custom_prompt_dict
            )
            model_response = anthropic.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                custom_prompt_dict=litellm.custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding, # for calculating input/output tokens
                api_key=api_key,
                logging_obj=logging, 
            )
            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(model_response, model, custom_llm_provider="anthropic", logging_obj=logging)
                return response
            response = model_response
        elif custom_llm_provider == "nlp_cloud":
            nlp_cloud_key = (
                api_key or litellm.nlp_cloud_key or get_secret("NLP_CLOUD_API_KEY") or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("NLP_CLOUD_API_BASE")
                or "https://api.nlpcloud.io/v1/gpu/"
            )

            model_response = nlp_cloud.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=nlp_cloud_key,
                logging_obj=logging
            )

            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(model_response, model, custom_llm_provider="nlp_cloud", logging_obj=logging)
                return response
            response = model_response
        elif custom_llm_provider == "aleph_alpha":
            aleph_alpha_key = (
                api_key or litellm.aleph_alpha_key or get_secret("ALEPH_ALPHA_API_KEY") or get_secret("ALEPHALPHA_API_KEY") or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("ALEPH_ALPHA_API_BASE")
                or "https://api.aleph-alpha.com/complete"
            )

            model_response = aleph_alpha.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                default_max_tokens_to_sample=litellm.max_tokens,
                api_key=aleph_alpha_key,
                logging_obj=logging # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
            )

            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(model_response, model, custom_llm_provider="aleph_alpha", logging_obj=logging)
                return response
            response = model_response
        elif custom_llm_provider == "cohere":
            cohere_key = (
                api_key
                or litellm.cohere_key
                or get_secret("COHERE_API_KEY")
                or get_secret("CO_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("COHERE_API_BASE")
                or "https://api.cohere.ai/v1/generate"
            )
            
            model_response = cohere.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=cohere_key,
                logging_obj=logging # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
            )

            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(model_response, model, custom_llm_provider="cohere", logging_obj=logging)
                return response
            response = model_response
        elif custom_llm_provider == "maritalk":
            maritalk_key = (
                api_key
                or litellm.maritalk_key
                or get_secret("MARITALK_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("MARITALK_API_BASE")
                or "https://chat.maritaca.ai/api/chat/inference"
            )
            
            model_response = maritalk.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=maritalk_key,
                logging_obj=logging 
            )

            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(model_response, model, custom_llm_provider="maritalk", logging_obj=logging)
                return response
            response = model_response
        elif ( 
            custom_llm_provider == "huggingface"
        ):
            custom_llm_provider = "huggingface"
            huggingface_key = (
                api_key
                or litellm.huggingface_key
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_API_KEY")
                or litellm.api_key
            )
            hf_headers = (
                headers
                or litellm.headers
            )

            custom_prompt_dict = (
                custom_prompt_dict
                or litellm.custom_prompt_dict
            )
            model_response = huggingface.completion(
                model=model,
                messages=messages,
                api_base=api_base, # type: ignore
                headers=hf_headers,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding, 
                api_key=huggingface_key, 
                acompletion=acompletion,
                logging_obj=logging,
                custom_prompt_dict=custom_prompt_dict
            )
            if "stream" in optional_params and optional_params["stream"] == True and acompletion is False:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response, model, custom_llm_provider="huggingface", logging_obj=logging
                )
                return response
            response = model_response
        elif custom_llm_provider == "oobabooga":
            custom_llm_provider = "oobabooga"
            model_response = oobabooga.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                api_base=api_base, # type: ignore
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=None,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging
            )
            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response, model, custom_llm_provider="oobabooga", logging_obj=logging
                )
                return response
            response = model_response
        elif custom_llm_provider == "openrouter":
            api_base = (
                api_base
                or litellm.api_base
                or  "https://openrouter.ai/api/v1"
            )

            api_key = (
                api_key or
                litellm.api_key or
                litellm.openrouter_key or
                get_secret("OPENROUTER_API_KEY") or 
                get_secret("OR_API_KEY")
            )

            openrouter_site_url = (
                get_secret("OR_SITE_URL")
                or "https://litellm.ai"
            )

            openrouter_app_name = (
                get_secret("OR_APP_NAME")
                or "liteLLM"
            )

            headers = (
                headers or
                litellm.headers or 
                {
                    "HTTP-Referer": openrouter_site_url,
                    "X-Title": openrouter_app_name,
                }
            )

            data = {
                "model": model, 
                "messages": messages,  
                **optional_params
            }
            ## LOGGING
            logging.pre_call(input=messages, api_key=openai.api_key, additional_args={"complete_input_dict": data, "headers": headers})
            ## COMPLETION CALL

            ## COMPLETION CALL
            response = openai_chat_completions.completion(
                model=model,
                messages=messages,
                headers=headers,
                api_key=api_key,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                logging_obj=logging, 
                acompletion=acompletion,
                timeout=timeout
            )
            ## LOGGING
            logging.post_call(
                input=messages, api_key=openai.api_key, original_response=response
            )
        elif custom_llm_provider == "together_ai" or ("togethercomputer" in model) or (model  in litellm.together_ai_models):
            custom_llm_provider = "together_ai"
            together_ai_key = (
                api_key
                or litellm.togetherai_api_key
                or get_secret("TOGETHER_AI_TOKEN")
                or get_secret("TOGETHERAI_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("TOGETHERAI_API_BASE")
                or "https://api.together.xyz/inference"
            )

            custom_prompt_dict = (
                custom_prompt_dict
                or litellm.custom_prompt_dict
            )
            
            model_response = together_ai.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=together_ai_key,
                logging_obj=logging,
                custom_prompt_dict=custom_prompt_dict
            )
            if "stream_tokens" in optional_params and optional_params["stream_tokens"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response, model, custom_llm_provider="together_ai", logging_obj=logging
                )
                return response
            response = model_response
        elif custom_llm_provider == "palm":
            palm_api_key = (
                api_key
                or get_secret("PALM_API_KEY")
                or litellm.api_key
            )
            
            # palm does not support streaming as yet :(
            model_response = palm.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=palm_api_key,
                logging_obj=logging
            )
            # fake palm streaming
            if "stream" in optional_params and optional_params["stream"] == True:
                # fake streaming for palm
                resp_string = model_response["choices"][0]["message"]["content"]
                response = CustomStreamWrapper(
                    resp_string, model, custom_llm_provider="palm", logging_obj=logging
                )
                return response
            response = model_response
        elif model in litellm.vertex_chat_models or model in litellm.vertex_code_chat_models or model in litellm.vertex_text_models or model in litellm.vertex_code_text_models:
            vertex_ai_project = (litellm.vertex_project 
                                 or get_secret("VERTEXAI_PROJECT"))
            vertex_ai_location = (litellm.vertex_location 
                                  or get_secret("VERTEXAI_LOCATION"))

            model_response = vertex_ai.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                vertex_location=vertex_ai_location,
                vertex_project=vertex_ai_project,
                logging_obj=logging
            )
            
            if "stream" in optional_params and optional_params["stream"] == True:
                response = CustomStreamWrapper(
                    model_response, model, custom_llm_provider="vertex_ai", logging_obj=logging
                    )
                return response
            response = model_response
        elif custom_llm_provider == "ai21":
            custom_llm_provider = "ai21"
            ai21_key = (
                api_key
                or litellm.ai21_key
                or os.environ.get("AI21_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("AI21_API_BASE")
                or "https://api.ai21.com/studio/v1/"
            )
        
            model_response = ai21.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=ai21_key,
                logging_obj=logging
            )
            
            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response, model, custom_llm_provider="ai21", logging_obj=logging
                )
                return response
            
            ## RESPONSE OBJECT
            response = model_response
        elif custom_llm_provider == "sagemaker":
            # boto3 reads keys from .env
            model_response = sagemaker.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging
            )
            if "stream" in optional_params and optional_params["stream"]==True: ## [BETA]
                # sagemaker does not support streaming as of now so we're faking streaming:
                # https://discuss.huggingface.co/t/streaming-output-text-when-deploying-on-sagemaker/39611
                # "SageMaker is currently not supporting streaming responses."
                
                # fake streaming for sagemaker
                resp_string = model_response["choices"][0]["message"]["content"]
                response = CustomStreamWrapper(
                    resp_string, model, custom_llm_provider="sagemaker", logging_obj=logging
                )
                return response

            ## RESPONSE OBJECT
            response = model_response
        elif custom_llm_provider == "bedrock":
            # boto3 reads keys from .env
            custom_prompt_dict = (
                custom_prompt_dict
                or litellm.custom_prompt_dict
            )
            model_response = bedrock.completion(
                model=model,
                messages=messages,
                custom_prompt_dict=litellm.custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
            )


            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    iter(model_response), model, custom_llm_provider="bedrock", logging_obj=logging
                )
                return response

            ## RESPONSE OBJECT
            response = model_response
        elif custom_llm_provider == "vllm":
            model_response = vllm.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging
            )

            if "stream" in optional_params and optional_params["stream"] == True: ## [BETA]
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response, model, custom_llm_provider="vllm", logging_obj=logging
                )
                return response

            ## RESPONSE OBJECT
            response = model_response
        elif custom_llm_provider == "ollama":
            api_base = (
                litellm.api_base or
                api_base or
                get_secret("OLLAMA_API_BASE") or 
                "http://localhost:11434"
                
            )
            custom_prompt_dict = (
                custom_prompt_dict
                or litellm.custom_prompt_dict
            )
            if model in custom_prompt_dict:
                # check if the model has a registered custom prompt
                model_prompt_details = custom_prompt_dict[model]
                prompt = custom_prompt(
                    role_dict=model_prompt_details["roles"], 
                    initial_prompt_value=model_prompt_details["initial_prompt_value"],  
                    final_prompt_value=model_prompt_details["final_prompt_value"], 
                    messages=messages
                )
            else:
                prompt = prompt_factory(model=model, messages=messages, custom_llm_provider=custom_llm_provider)
            ## LOGGING
            if kwargs.get('acompletion', False) == True:    
                if optional_params.get("stream", False) == True:
                # assume all ollama responses are streamed
                    async_generator = ollama.async_get_ollama_response_stream(api_base, model, prompt, optional_params, logging_obj=logging)
                    return async_generator

            generator = ollama.get_ollama_response_stream(api_base, model, prompt, optional_params, logging_obj=logging)
            if optional_params.get("stream", False) == True:
                # assume all ollama responses are streamed
                response = CustomStreamWrapper(
                        generator, model, custom_llm_provider="ollama", logging_obj=logging
                )
                return response
            else:
                response_string = ""
                for chunk in generator:
                    response_string+=chunk['content']
            
            ## RESPONSE OBJECT
            model_response["choices"][0]["message"]["content"] = response_string
            model_response["created"] = time.time()
            model_response["model"] = "ollama/" + model
            prompt_tokens = len(encoding.encode(prompt)) # type: ignore
            completion_tokens = len(encoding.encode(response_string))
            model_response["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            response = model_response
        elif (
            custom_llm_provider == "baseten"
            or litellm.api_base == "https://app.baseten.co"
        ):
            custom_llm_provider = "baseten"
            baseten_key = (
                api_key or litellm.baseten_key or os.environ.get("BASETEN_API_KEY") or litellm.api_key
            )

            model_response = baseten.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding, 
                api_key=baseten_key, 
                logging_obj=logging
            )
            if inspect.isgenerator(model_response) or ("stream" in optional_params and optional_params["stream"] == True):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response, model, custom_llm_provider="baseten", logging_obj=logging
                )
                return response
            response = model_response
        elif (
            custom_llm_provider == "petals"
            or model in litellm.petals_models
        ):
            api_base = (
                litellm.api_base or
                api_base 
            )

            custom_llm_provider = "petals"
            model_response = petals.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding, 
                logging_obj=logging
            )
            if stream==True: ## [BETA]
                # Fake streaming for petals
                resp_string = model_response["choices"][0]["message"]["content"]
                response = CustomStreamWrapper(
                    resp_string, model, custom_llm_provider="petals", logging_obj=logging
                )
                return response
            response = model_response
        elif (
            custom_llm_provider == "custom"
            ):
            import requests

            url = (
                litellm.api_base or
                api_base or
                ""
            )
            if url == None or url == "":
                raise ValueError("api_base not set. Set api_base or litellm.api_base for custom endpoints")

            """
            assume input to custom LLM api bases follow this format:
            resp = requests.post(
                api_base, 
                json={
                    'model': 'meta-llama/Llama-2-13b-hf', # model name
                    'params': {
                        'prompt': ["The capital of France is P"],
                        'max_tokens': 32,
                        'temperature': 0.7,
                        'top_p': 1.0,
                        'top_k': 40,
                    }
                }
            )

            """
            prompt = " ".join([message["content"] for message in messages]) # type: ignore
            resp = requests.post(url, json={
                'model': model,
                'params': {
                    'prompt': [prompt],
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': kwargs.get('top_k', 40),
                }
            })
            response_json = resp.json()
            """
            assume all responses from custom api_bases of this format:
            {
                'data': [
                    {
                        'prompt': 'The capital of France is P',
                        'output': ['The capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France'],
                        'params': {'temperature': 0.7, 'top_k': 40, 'top_p': 1}}],
                        'message': 'ok'
                    }
                ]
            }
            """
            string_response = response_json['data'][0]['output'][0]
            ## RESPONSE OBJECT
            model_response["choices"][0]["message"]["content"] = string_response
            model_response["created"] = time.time()
            model_response["model"] = model
            response = model_response
        else:
            raise ValueError(
                f"Unable to map your input to a model. Check your input - {args}"
            )
        return response
    except Exception as e:
        ## Map to OpenAI Exception
        raise exception_type(
                model=model, custom_llm_provider=custom_llm_provider, original_exception=e, completion_kwargs=args,
            )


def completion_with_retries(*args, **kwargs):
    """
    Executes a litellm.completion() with 3 retries
    """
    try:
        import tenacity
    except Exception as e:
        raise Exception(f"tenacity import failed please run `pip install tenacity`. Error{e}")
    
    num_retries = kwargs.pop("num_retries", 3)
    retry_strategy = kwargs.pop("retry_strategy", "constant_retry")
    original_function = kwargs.pop("original_function", completion)
    if retry_strategy == "constant_retry": 
        retryer = tenacity.Retrying(stop=tenacity.stop_after_attempt(num_retries), reraise=True)
    elif retry_strategy == "exponential_backoff_retry": 
        retryer = tenacity.Retrying(wait=tenacity.wait_exponential(multiplier=1, max=10), stop=tenacity.stop_after_attempt(num_retries), reraise=True)
    return retryer(original_function, *args, **kwargs)

async def acompletion_with_retries(*args, **kwargs):
    """
    Executes a litellm.completion() with 3 retries
    """
    try:
        import tenacity
    except Exception as e:
        raise Exception(f"tenacity import failed please run `pip install tenacity`. Error{e}")
    
    num_retries = kwargs.pop("num_retries", 3)
    retry_strategy = kwargs.pop("retry_strategy", "constant_retry")
    original_function = kwargs.pop("original_function", completion)
    if retry_strategy == "constant_retry": 
        retryer = tenacity.Retrying(stop=tenacity.stop_after_attempt(num_retries), reraise=True)
    elif retry_strategy == "exponential_backoff_retry": 
        retryer = tenacity.Retrying(wait=tenacity.wait_exponential(multiplier=1, max=10), stop=tenacity.stop_after_attempt(num_retries), reraise=True)
    return await retryer(original_function, *args, **kwargs)



def batch_completion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: List = [],
    functions: List = [],
    function_call: str = "",  # optional params
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stop=None,
    max_tokens: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float]=None,
    logit_bias: dict = {},
    user: str = "",
    deployment_id = None,
    request_timeout: Optional[int] = None,
    # Optional liteLLM function params
    **kwargs):
    """
    Batch litellm.completion function for a given model.

    Args:
        model (str): The model to use for generating completions.
        messages (List, optional): List of messages to use as input for generating completions. Defaults to [].
        functions (List, optional): List of functions to use as input for generating completions. Defaults to [].
        function_call (str, optional): The function call to use as input for generating completions. Defaults to "".
        temperature (float, optional): The temperature parameter for generating completions. Defaults to None.
        top_p (float, optional): The top-p parameter for generating completions. Defaults to None.
        n (int, optional): The number of completions to generate. Defaults to None.
        stream (bool, optional): Whether to stream completions or not. Defaults to None.
        stop (optional): The stop parameter for generating completions. Defaults to None.
        max_tokens (float, optional): The maximum number of tokens to generate. Defaults to None.
        presence_penalty (float, optional): The presence penalty for generating completions. Defaults to None.
        frequency_penalty (float, optional): The frequency penalty for generating completions. Defaults to None.
        logit_bias (dict, optional): The logit bias for generating completions. Defaults to {}.
        user (str, optional): The user string for generating completions. Defaults to "".
        deployment_id (optional): The deployment ID for generating completions. Defaults to None.
        request_timeout (int, optional): The request timeout for generating completions. Defaults to None.

    Returns:
        list: A list of completion results.
    """
    args = locals()
    batch_messages = messages
    completions = []
    model = model
    custom_llm_provider = None
    if model.split("/", 1)[0] in litellm.provider_list:
        custom_llm_provider = model.split("/", 1)[0]
        model = model.split("/", 1)[1]
    if custom_llm_provider == "vllm":
        optional_params = get_optional_params(
            functions=functions,
            function_call=function_call,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            # params to identify the model
            model=model,
            custom_llm_provider=custom_llm_provider
        )
        results = vllm.batch_completions(model=model, messages=batch_messages, custom_prompt_dict=litellm.custom_prompt_dict, optional_params=optional_params)
    # all non VLLM models for batch completion models 
    else:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        with ThreadPoolExecutor(max_workers=100) as executor:
            for sub_batch in chunks(batch_messages, 100):
                for message_list in sub_batch:
                    kwargs_modified = args.copy()
                    kwargs_modified["messages"] = message_list
                    original_kwargs = {}
                    if "kwargs" in kwargs_modified:
                        original_kwargs = kwargs_modified.pop("kwargs")
                    future = executor.submit(completion, **kwargs_modified, **original_kwargs)
                    completions.append(future)

        # Retrieve the results from the futures
        results = [future.result() for future in completions]
    return results

# send one request to multiple models
# return as soon as one of the llms responds
def batch_completion_models(*args, **kwargs):
    """
    Send a request to multiple language models concurrently and return the response
    as soon as one of the models responds.

    Args:
        *args: Variable-length positional arguments passed to the completion function.
        **kwargs: Additional keyword arguments:
            - models (str or list of str): The language models to send requests to.
            - Other keyword arguments to be passed to the completion function.

    Returns:
        str or None: The response from one of the language models, or None if no response is received.

    Note:
        This function utilizes a ThreadPoolExecutor to parallelize requests to multiple models.
        It sends requests concurrently and returns the response from the first model that responds.
    """
    import concurrent
    if "model" in kwargs:
        kwargs.pop("model")
    if "models" in kwargs:
        models = kwargs["models"]
        kwargs.pop("models")
        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
            for model in models:
                futures[model] = executor.submit(completion, *args, model=model, **kwargs)

            for model, future in sorted(futures.items(), key=lambda x: models.index(x[0])):
                if future.result() is not None:
                    return future.result()
    elif "deployments" in kwargs: 
        deployments = kwargs["deployments"]
        kwargs.pop("deployments")
        kwargs.pop("model_list")
        nested_kwargs = kwargs.pop("kwargs", {})
        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(deployments)) as executor:
            for deployment in deployments:
                for key in kwargs.keys(): 
                    if key not in deployment: # don't override deployment values e.g. model name, api base, etc. 
                        deployment[key] = kwargs[key]
                kwargs = {**deployment, **nested_kwargs}
                futures[deployment["model"]] = executor.submit(completion, **kwargs)

            while futures:
                # wait for the first returned future
                print_verbose("\n\n waiting for next result\n\n")
                done, _ = concurrent.futures.wait(futures.values(), return_when=concurrent.futures.FIRST_COMPLETED)
                print_verbose(f"done list\n{done}")
                for future in done:
                    try:
                        result = future.result()
                        return result
                    except Exception as e:
                        # if model 1 fails, continue with response from model 2, model3
                        print_verbose(f"\n\ngot an exception, ignoring, removing from futures")
                        print_verbose(futures)
                        new_futures = {}
                        for key, value in futures.items():
                            if future == value:
                                print_verbose(f"removing key{key}")
                                continue
                            else:
                                new_futures[key] = value
                        futures = new_futures
                        print_verbose(f"new futures{futures}")
                        continue

                
                print_verbose("\n\ndone looping through futures\n\n")
                print_verbose(futures)

    return None  # If no response is received from any model

def batch_completion_models_all_responses(*args, **kwargs):
    """
    Send a request to multiple language models concurrently and return a list of responses
    from all models that respond.

    Args:
        *args: Variable-length positional arguments passed to the completion function.
        **kwargs: Additional keyword arguments:
            - models (str or list of str): The language models to send requests to.
            - Other keyword arguments to be passed to the completion function.

    Returns:
        list: A list of responses from the language models that responded.

    Note:
        This function utilizes a ThreadPoolExecutor to parallelize requests to multiple models.
        It sends requests concurrently and collects responses from all models that respond.
    """
    import concurrent.futures

    # ANSI escape codes for colored output
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    if "model" in kwargs:
        kwargs.pop("model")
    if "models" in kwargs:
        models = kwargs["models"]
        kwargs.pop("models")

    responses = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        for idx, model in enumerate(models):
            future = executor.submit(completion, *args, model=model, **kwargs)
            if future.result() is not None:
                responses.append(future.result())

    return responses

### EMBEDDING ENDPOINTS ####################

async def aembedding(*args, **kwargs):
    """
    Asynchronously calls the `embedding` function with the given arguments and keyword arguments.

    Parameters:
    - `args` (tuple): Positional arguments to be passed to the `embedding` function.
    - `kwargs` (dict): Keyword arguments to be passed to the `embedding` function.

    Returns:
    - `response` (Any): The response returned by the `embedding` function.
    """
    loop = asyncio.get_event_loop()

    # Use a partial function to pass your keyword arguments
    func = partial(embedding, *args, **kwargs)

    # Add the context to the function
    ctx = contextvars.copy_context()
    func_with_context = partial(ctx.run, func)

    # Call the synchronous function using run_in_executor
    response =  await loop.run_in_executor(None, func_with_context)
    return response

@client
def embedding(
    model, 
    input=[], 
    # Optional params
    azure=False, 
    force_timeout=60, 
    litellm_call_id=None, 
    litellm_logging_obj=None,
    logger_fn=None, 
    # set api_base, api_version, api_key
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    api_type: Optional[str] = None,
    caching: bool=False,
    custom_llm_provider=None,
    **kwargs
):
    """
    Embedding function that calls an API to generate embeddings for the given input.

    Parameters:
    - model: The embedding model to use.
    - input: The input for which embeddings are to be generated.
    - azure: A boolean indicating whether to use the Azure API for embedding.
    - force_timeout: The timeout value for the API call.
    - litellm_call_id: The call ID for litellm logging.
    - litellm_logging_obj: The litellm logging object.
    - logger_fn: The logger function.
    - api_base: Optional. The base URL for the API.
    - api_version: Optional. The version of the API.
    - api_key: Optional. The API key to use.
    - api_type: Optional. The type of the API.
    - caching: A boolean indicating whether to enable caching.
    - custom_llm_provider: The custom llm provider.

    Returns:
    - response: The response received from the API call.

    Raises:
    - exception_type: If an exception occurs during the API call.
    """
    model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider, api_base=api_base)
    try:
        response = None
        logging = litellm_logging_obj
        logging.update_environment_variables(model=model, user="", optional_params={}, litellm_params={"force_timeout": force_timeout, "azure": azure, "litellm_call_id": litellm_call_id, "logger_fn": logger_fn})
        if azure == True or custom_llm_provider == "azure":
            # azure configs
            api_type = get_secret("AZURE_API_TYPE") or "azure"

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("AZURE_API_BASE")
            )

            api_version = (
                api_version or
                litellm.api_version or
                get_secret("AZURE_API_VERSION")
            )

            azure_ad_token = (
                kwargs.pop("azure_ad_token", None) or
                get_secret("AZURE_AD_TOKEN")
            )

            api_key = (
                api_key or
                litellm.api_key or
                litellm.azure_key or
                get_secret("AZURE_API_KEY")
            )
            ## EMBEDDING CALL
            response = azure_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                api_version=api_version,
                azure_ad_token=azure_ad_token,
                logging_obj=logging,
                model_response=EmbeddingResponse(), 
                optional_params=kwargs
            )
        elif model in litellm.open_ai_embedding_models:
            api_base = (
                api_base
                or litellm.api_base
                or get_secret("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            openai.organization = (
                litellm.organization
                or get_secret("OPENAI_ORGANIZATION")
                or None # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
            )
            # set API KEY
            api_key = (
                api_key or
                litellm.api_key or
                litellm.openai_key or
                get_secret("OPENAI_API_KEY")
            )
            api_type = "openai"
            api_version = None


            ## EMBEDDING CALL
            response = openai_chat_completions.embedding(
                model=model,
                input=input,
                api_base=api_base,
                api_key=api_key,
                logging_obj=logging,
                model_response=EmbeddingResponse(), 
                optional_params=kwargs
            )
        elif model in litellm.cohere_embedding_models:
            cohere_key = (
                api_key
                or litellm.cohere_key
                or get_secret("COHERE_API_KEY")
                or get_secret("CO_API_KEY")
                or litellm.api_key
            )
            response = cohere.embedding(
                model=model,
                input=input,
                optional_params=kwargs,
                encoding=encoding,
                api_key=cohere_key,
                logging_obj=logging,
                model_response= EmbeddingResponse()

            )
        elif custom_llm_provider == "huggingface":
            api_key = (
                api_key
                or litellm.huggingface_key
                or get_secret("HUGGINGFACE_API_KEY")
                or litellm.api_key
            )
            response = huggingface.embedding(
                model=model,
                input=input,
                encoding=encoding,
                api_key=api_key,
                api_base=api_base,
                logging_obj=logging,
                model_response= EmbeddingResponse()
            )
        elif custom_llm_provider == "bedrock":
            response = bedrock.embedding(
                model=model,
                input=input,
                encoding=encoding,
                logging_obj=logging,
                optional_params=kwargs,
                model_response= EmbeddingResponse()
            )
        else:
            args = locals()
            raise ValueError(f"No valid embedding model args passed in - {args}")
        return response
    except Exception as e:
        ## LOGGING
        logging.post_call(
            input=input,
            api_key=openai.api_key,
            original_response=str(e),
        )
        ## Map to OpenAI Exception
        raise exception_type(
            model=model,
            original_exception=e,
            custom_llm_provider="azure" if azure == True else None,
        )


###### Text Completion ################
def text_completion(
    prompt: Union[str, List[Union[str, List[Union[str, List[int]]]]]], # Required: The prompt(s) to generate completions for.
    model: Optional[str]=None,                 # Optional: either `model` or `engine` can be set
    best_of: Optional[int] = None,   # Optional: Generates best_of completions server-side.
    echo: Optional[bool] = None,  # Optional: Echo back the prompt in addition to the completion.
    frequency_penalty: Optional[float] = None, # Optional: Penalize new tokens based on their existing frequency.
    logit_bias: Optional[Dict[int, int]] = None, # Optional: Modify the likelihood of specified tokens.
    logprobs: Optional[int] = None, # Optional: Include the log probabilities on the most likely tokens.
    max_tokens: Optional[int] = None, # Optional: The maximum number of tokens to generate in the completion.
    n: Optional[int] = None,         # Optional: How many completions to generate for each prompt.
    presence_penalty: Optional[float] = None, # Optional: Penalize new tokens based on whether they appear in the text so far.
    stop: Optional[Union[str, List[str]]] = None, # Optional: Sequences where the API will stop generating further tokens.
    stream: Optional[bool] = None, # Optional: Whether to stream back partial progress.
    suffix: Optional[str] = None,   # Optional: The suffix that comes after a completion of inserted text.
    temperature: Optional[float] = None, # Optional: Sampling temperature to use.
    top_p: Optional[float] = None,     # Optional: Nucleus sampling parameter.
    user: Optional[str] = None,     # Optional: A unique identifier representing your end-user.

    # set api_base, api_version, api_key
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    model_list: Optional[list] = None, # pass in a list of api_base,keys, etc. 

    # Optional liteLLM function params
    custom_llm_provider: Optional[str] = None,
    *args, 
    **kwargs
):
    global print_verbose
    import copy
    """
    Generate text completions using the OpenAI API.

    Args:
        model (str): ID of the model to use.
        prompt (Union[str, List[Union[str, List[Union[str, List[int]]]]]): The prompt(s) to generate completions for.
        best_of (Optional[int], optional): Generates best_of completions server-side. Defaults to 1.
        echo (Optional[bool], optional): Echo back the prompt in addition to the completion. Defaults to False.
        frequency_penalty (Optional[float], optional): Penalize new tokens based on their existing frequency. Defaults to 0.
        logit_bias (Optional[Dict[int, int]], optional): Modify the likelihood of specified tokens. Defaults to None.
        logprobs (Optional[int], optional): Include the log probabilities on the most likely tokens. Defaults to None.
        max_tokens (Optional[int], optional): The maximum number of tokens to generate in the completion. Defaults to 16.
        n (Optional[int], optional): How many completions to generate for each prompt. Defaults to 1.
        presence_penalty (Optional[float], optional): Penalize new tokens based on whether they appear in the text so far. Defaults to 0.
        stop (Optional[Union[str, List[str]]], optional): Sequences where the API will stop generating further tokens. Defaults to None.
        stream (Optional[bool], optional): Whether to stream back partial progress. Defaults to False.
        suffix (Optional[str], optional): The suffix that comes after a completion of inserted text. Defaults to None.
        temperature (Optional[float], optional): Sampling temperature to use. Defaults to 1.
        top_p (Optional[float], optional): Nucleus sampling parameter. Defaults to 1.
        user (Optional[str], optional): A unique identifier representing your end-user.
    Returns:
        TextCompletionResponse: A response object containing the generated completion and associated metadata.

    Example:
        Your example of how to use this function goes here.
    """
    if "engine" in  kwargs:
        if model==None:
            # only use engine when model not passed
            model = kwargs["engine"]
        kwargs.pop("engine")

    text_completion_response = TextCompletionResponse()

    optional_params: Dict[str, Any] = {}
    # default values for all optional params are none, litellm only passes them to the llm when they are set to non None values
    if best_of is not None:
        optional_params["best_of"] = best_of
    if echo is not None:
        optional_params["echo"] = echo
    if frequency_penalty is not None:
        optional_params["frequency_penalty"] = frequency_penalty
    if logit_bias is not None:
        optional_params["logit_bias"] = logit_bias
    if logprobs is not None:
        optional_params["logprobs"] = logprobs
    if max_tokens is not None:
        optional_params["max_tokens"] = max_tokens
    if n is not None:
        optional_params["n"] = n
    if presence_penalty is not None:
        optional_params["presence_penalty"] = presence_penalty
    if stop is not None:
        optional_params["stop"] = stop
    if stream is not None:
        optional_params["stream"] = stream
    if suffix is not None:
        optional_params["suffix"] = suffix
    if temperature is not None:
        optional_params["temperature"] = temperature
    if top_p is not None:
        optional_params["top_p"] = top_p
    if user is not None:
        optional_params["user"] = user
    if api_base is not None:
        optional_params["api_base"] = api_base
    if api_version is not None:
        optional_params["api_version"] = api_version
    if api_key is not None:
        optional_params["api_key"] = api_key
    if custom_llm_provider is not None:
        optional_params["custom_llm_provider"] = custom_llm_provider

    # get custom_llm_provider
    _, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider, api_base=api_base) # type: ignore

    if custom_llm_provider == "huggingface":
        # if echo == True, for TGI llms we need to set top_n_tokens to 3
        if echo == True:
            # for tgi llms
            if "top_n_tokens" not in kwargs:
                kwargs["top_n_tokens"] = 3

        # processing prompt - users can pass raw tokens to OpenAI Completion()
        if type(prompt) == list:
            import concurrent.futures
            tokenizer = tiktoken.encoding_for_model("text-davinci-003")
            ## if it's a 2d list - each element in the list is a text_completion() request
            if len(prompt) > 0 and type(prompt[0]) == list:
                responses = [None for x in prompt] # init responses 
                def process_prompt(i, individual_prompt):
                    decoded_prompt = tokenizer.decode(individual_prompt)
                    all_params = {**kwargs, **optional_params}
                    response = text_completion(
                        model=model,
                        prompt=decoded_prompt,
                        num_retries=3,# ensure this does not fail for the batch
                        *args,
                        **all_params,
                    )
                    #print(response)
                    text_completion_response["id"] = response.get("id", None)
                    text_completion_response["object"] = "text_completion"
                    text_completion_response["created"] = response.get("created", None)
                    text_completion_response["model"] = response.get("model", None)
                    return response["choices"][0]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_prompt, i, individual_prompt) for i, individual_prompt in enumerate(prompt)]
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        responses[i] = future.result()
                    text_completion_response.choices = responses 

                return text_completion_response
    # else:
    # check if non default values passed in for best_of, echo, logprobs, suffix 
    # these are the params supported by Completion() but not ChatCompletion
    
    # default case, non OpenAI requests go through here
    messages = [{"role": "system", "content": prompt}]
    kwargs.pop("prompt", None)
    response = completion(
        model = model,
        messages=messages,
        *args,
        **kwargs,
        **optional_params,
    )
    if stream == True or kwargs.get("stream", False) == True:
        response = TextCompletionStreamWrapper(completion_stream=response, model=model)
        return response

    transformed_logprobs = None
    # only supported for TGI models
    try:
        raw_response = response._hidden_params.get("original_response", None)
        transformed_logprobs = litellm.utils.transform_logprobs(raw_response)
    except Exception as e:
        print_verbose(f"LiteLLM non blocking exception: {e}")
    text_completion_response["id"] = response.get("id", None)
    text_completion_response["object"] = "text_completion"
    text_completion_response["created"] = response.get("created", None)
    text_completion_response["model"] = response.get("model", None)
    text_choices = TextChoices()
    text_choices["text"] = response["choices"][0]["message"]["content"]
    text_choices["index"] = response["choices"][0]["index"]
    text_choices["logprobs"] = transformed_logprobs
    text_choices["finish_reason"] = response["choices"][0]["finish_reason"]
    text_completion_response["choices"] = [text_choices]
    text_completion_response["usage"] = response.get("usage", None)
    return text_completion_response

##### Moderation #######################
def moderation(input: str, api_key: Optional[str]=None):
    # only supports open ai for now
    api_key = (
                api_key or
                litellm.api_key or
                litellm.openai_key or
                get_secret("OPENAI_API_KEY")
            )
    openai.api_key = api_key
    openai.api_type = "open_ai" # type: ignore
    openai.api_version = None
    openai.base_url = "https://api.openai.com/v1/"
    response = openai.moderations.create(input=input)
    return response

####### HELPER FUNCTIONS ################
## Set verbose to true -> ```litellm.set_verbose = True```
def print_verbose(print_statement):
    if litellm.set_verbose:
        print(print_statement) # noqa

def config_completion(**kwargs):
    if litellm.config_path != None:
        config_args = read_config_args(litellm.config_path)
        # overwrite any args passed in with config args
        return completion(**kwargs, **config_args)
    else:
        raise ValueError(
            "No config path set, please set a config path using `litellm.config_path = 'path/to/config.json'`"
        )

def stream_chunk_builder(chunks: list):
    id = chunks[0]["id"]
    object = chunks[0]["object"]
    created = chunks[0]["created"]
    model = chunks[0]["model"]
    role = chunks[0]["choices"][0]["delta"]["role"]
    finish_reason = chunks[-1]["choices"][0]["finish_reason"]
    
    # Initialize the response dictionary
    response = {
        "id": id,
        "object": object,
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": role,
                    "content": ""
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # Modify as needed
            "completion_tokens": 0,  # Modify as needed
            "total_tokens": 0  # Modify as needed
        }
    }

    # Extract the "content" strings from the nested dictionaries within "choices"
    content_list = []
    combined_content = ""

    if "function_call" in chunks[0]["choices"][0]["delta"] and chunks[0]["choices"][0]["delta"]["function_call"] is not None:
        argument_list = []
        delta = chunks[0]["choices"][0]["delta"]
        function_call = delta.get("function_call", "")
        function_call_name = function_call.get("name", "")

        message = response["choices"][0]["message"]
        message["function_call"] = {}
        message["function_call"]["name"] = function_call_name

        for chunk in chunks:
            choices = chunk["choices"]
            for choice in choices:
                delta = choice.get("delta", {})
                function_call = delta.get("function_call", "")
                
                # Check if a function call is present
                if function_call:
                    # Now, function_call is expected to be a dictionary
                    arguments = function_call.get("arguments", "")
                    argument_list.append(arguments)

        combined_arguments = "".join(argument_list)
        response["choices"][0]["message"]["content"] = None
        response["choices"][0]["message"]["function_call"]["arguments"] = combined_arguments
    else:
        for chunk in chunks:
            choices = chunk["choices"]
            for choice in choices:
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                if content == None:
                    continue # openai v1.0.0 sets content = None for chunks
                content_list.append(content)

        # Combine the "content" strings into a single string
        combined_content = "".join(content_list)

        # Update the "content" field within the response dictionary
        response["choices"][0]["message"]["content"] = combined_content


    # # Update usage information if needed
    response["usage"]["completion_tokens"] = litellm.utils.token_counter(model=model, text=combined_content)
    return response
