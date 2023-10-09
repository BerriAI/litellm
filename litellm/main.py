# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you ! We ❤️ you! - Krrish & Ishaan

import os, openai, sys, json, inspect
from typing import Any
from functools import partial
import dotenv, traceback, random, asyncio, time, contextvars
from copy import deepcopy
import litellm
from litellm import (  # type: ignore
    client,
    exception_type,
    timeout,
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
    vertex_ai)
from .llms.prompt_templates.factory import prompt_factory, custom_prompt, function_call_prompt
import tiktoken
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Dict

encoding = tiktoken.get_encoding("cl100k_base")
from litellm.utils import (
    get_secret,
    CustomStreamWrapper,
    ModelResponse,
    EmbeddingResponse,
    read_config_args,
    RateLimitManager
)

####### ENVIRONMENT VARIABLES ###################
dotenv.load_dotenv()  # Loading env variables using dotenv

####### COMPLETION ENDPOINTS ################

async def acompletion(*args, **kwargs):
    """
    Asynchronously perform a completion() using the any LiteLLM model (ex gpt-3.5-turbo, claude-2)

    This function takes the same arguments as the 'completion' function and is used for asynchronous completion requests.

    Parameters:
        *args: Positional arguments to pass to the 'litellm.completion' function.
        **kwargs: Keyword arguments to pass to the 'litellm.completion' function.

    Returns:
        The completion response, either as a litellm.ModelResponse Object or an async generator if 'stream' is set to True.

    Note:
        - This function uses asynchronous programming to perform completions.
        - It leverages the 'loop.run_in_executor' method to execute the synchronous 'completion' function.
        - If 'stream' is set to True in kwargs, the function returns an async generator.
    """
    loop = asyncio.get_event_loop()

    # Use a partial function to pass your keyword arguments
    kwargs["acompletion"] = True
    func = partial(completion, *args, **kwargs)

    # Add the context to the function
    ctx = contextvars.copy_context()
    func_with_context = partial(ctx.run, func)

    # Call the synchronous function using run_in_executor
    response =  await loop.run_in_executor(None, func_with_context)
    if kwargs.get("stream", False): # return an async generator
        # do not change this
        # for stream = True, always return an async generator
        # See OpenAI acreate https://github.com/openai/openai-python/blob/5d50e9e3b39540af782ca24e65c290343d86e1a9/openai/api_resources/abstract/engine_api_resource.py#L193
        return(
            line
            async for line in response
        )
    else:
        return response

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
@timeout(  # type: ignore
    600
)  ## set timeouts, in case calls hang (e.g. Azure) - default is 600s, override with `force_timeout`
def completion(
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

    # set api_base, api_version, api_key
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,

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

        LITELLM Specific Params
        mock_response (str, optional): If provided, return a mock completion response for testing or debugging purposes (default is None).
        force_timeout (int, optional): The maximum execution time in seconds for the completion request (default is 600).
        custom_llm_provider (str, optional): Used for Non-OpenAI LLMs, Example usage for bedrock, set model="amazon.titan-tg1-large" and custom_llm_provider="bedrock"
    Returns:
        ModelResponse: A response object containing the generated completion and associated metadata.

    Note:
        - This function is used to perform completions() using the specified language model.
        - It supports various optional parameters for customizing the completion behavior.
        - If 'mock_response' is provided, a mock completion response is returned for testing or debugging.
    """
    ######### unpacking kwargs #####################
    args = locals()
    return_async = kwargs.get('return_async', False)
    mock_response = kwargs.get('mock_response', None)
    force_timeout= kwargs.get('force_timeout', 600)
    logger_fn = kwargs.get('logger_fn', None)
    verbose = kwargs.get('verbose', False)
    custom_llm_provider = kwargs.get('custom_llm_provider', None)
    litellm_logging_obj = kwargs.get('litellm_logging_obj', None)
    id = kwargs.get('id', None)
    metadata = kwargs.get('metadata', None)
    fallbacks = kwargs.get('fallbacks', [])
    ######## end of unpacking kwargs ###########
    openai_params = ["functions", "function_call", "temperature", "temperature", "top_p", "n", "stream", "stop", "max_tokens", "presence_penalty", "frequency_penalty", "logit_bias", "user", "request_timeout", "api_base", "api_version", "api_key"]
    litellm_params = ["metadata", "acompletion", "caching", "return_async", "mock_response", "api_key", "api_version", "api_base", "force_timeout", "logger_fn", "verbose", "custom_llm_provider", "litellm_logging_obj", "litellm_call_id", "use_client", "id", "metadata", "fallbacks", "azure"]
    default_params = openai_params + litellm_params
    non_default_params = {k: v for k,v in kwargs.items() if k not in default_params} # model-specific params - pass them straight to the model/provider
    if mock_response:
        return mock_completion(model, messages, stream=stream, mock_response=mock_response)
    try:
        logging = litellm_logging_obj
        if fallbacks != []:
            return completion_with_fallbacks(**args)
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
        model, custom_llm_provider = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
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
                request_timeout=request_timeout,
                deployment_id=deployment_id,
                # params to identify the model
                model=model,
                custom_llm_provider=custom_llm_provider,
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

            ## LOAD CONFIG - if set
            config=litellm.AzureOpenAIConfig.get_config()
            for k, v in config.items():
                if k not in optional_params: # completion(top_k=3) > azure_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ## LOGGING
            logging.pre_call(
                input=messages,
                api_key=api_key,
                additional_args={
                    "headers": litellm.headers,
                    "api_version": api_version,
                    "api_base": api_base,
                },
            )
            ## COMPLETION CALL
            response = openai.ChatCompletion.create(
                engine=model,
                messages=messages,
                headers=litellm.headers,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                api_type=api_type,
                **optional_params,
            )
            if "stream" in optional_params and optional_params["stream"] == True:
                response = CustomStreamWrapper(response, model, custom_llm_provider="openai", logging_obj=logging)
                return response
            ## LOGGING
            logging.post_call(
                input=messages,
                api_key=api_key,
                original_response=response,
                additional_args={
                    "headers": litellm.headers,
                    "api_version": api_version,
                    "api_base": api_base,
                },
            )
        elif (
            model in litellm.open_ai_chat_completion_models
            or custom_llm_provider == "custom_openai"
            or custom_llm_provider == "openai"
            or "ft:gpt-3.5-turbo" in model  # finetuned gpt-3.5-turbo
        ):  # allow user to make an openai call with a custom base
            # note: if a user sets a custom base - we should ensure this works
            # allow for the setting of dynamic and stateful api-bases
            api_base = (
                api_base
                or litellm.api_base
                or get_secret("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            if litellm.organization:
                openai.organization = litellm.organization
            # set API KEY
            api_key = (
                api_key or
                litellm.api_key or
                litellm.openai_key or
                get_secret("OPENAI_API_KEY")
            )

            ## LOAD CONFIG - if set
            config=litellm.OpenAIConfig.get_config()
            for k, v in config.items():
                if k not in optional_params: # completion(top_k=3) > openai_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ## LOGGING
            logging.pre_call(
                input=messages,
                api_key=api_key,
                additional_args={"headers": litellm.headers, "api_base": api_base},
            )
            ## COMPLETION CALL
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    headers=litellm.headers, # None by default
                    api_base=api_base, # thread safe setting base, key, api_version
                    api_key=api_key,
                    api_type="openai",
                    api_version=api_version, # default None
                    **optional_params,
                )
            except Exception as e:
                ## LOGGING - log the original exception returned
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=str(e),
                    additional_args={"headers": litellm.headers},
                )
                raise e
            
            if "stream" in optional_params and optional_params["stream"] == True:
                response = CustomStreamWrapper(response, model, custom_llm_provider="openai", logging_obj=logging)
                return response
            ## LOGGING
            logging.post_call(
                input=messages,
                api_key=api_key,
                original_response=response,
                additional_args={"headers": litellm.headers},
            )
        elif (
            model in litellm.open_ai_text_completion_models
            or "ft:babbage-002" in model
            or "ft:davinci-002" in model  # support for finetuned completion models
            # NOTE: Do NOT add custom_llm_provider == "openai". 
            # this will break hosted vllm/proxy calls. 
            # see: https://docs.litellm.ai/docs/providers/vllm#calling-hosted-vllm-server. 
            # VLLM expects requests to call openai.ChatCompletion we need those requests to always 
            # call openai.ChatCompletion
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

            ## LOAD CONFIG - if set
            config=litellm.OpenAITextCompletionConfig.get_config()
            for k, v in config.items():
                if k not in optional_params: # completion(top_k=3) > openai_text_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v


            if litellm.organization:
                openai.organization = litellm.organization
            prompt = " ".join([message["content"] for message in messages])
            ## LOGGING
            logging.pre_call(
                input=prompt,
                api_key=api_key,
                additional_args={
                    "openai_organization": litellm.organization,
                    "headers": litellm.headers,
                    "api_base": api_base,
                    "api_type": openai.api_type,
                },
            )
            ## COMPLETION CALL
            response = openai.Completion.create(
                model=model, 
                prompt=prompt,
                headers=litellm.headers,
                api_key = api_key,
                api_base=api_base,
                **optional_params
            )
            if "stream" in optional_params and optional_params["stream"] == True:
                response = CustomStreamWrapper(response, model, custom_llm_provider="text-completion-openai", logging_obj=logging)
                return response
            ## LOGGING
            logging.post_call(
                input=prompt,
                api_key=api_key,
                original_response=response,
                additional_args={
                    "openai_organization": litellm.organization,
                    "headers": litellm.headers,
                    "api_base": openai.api_base,
                    "api_type": openai.api_type,
                },
            )
            ## RESPONSE OBJECT
            completion_response = response["choices"][0]["text"]
            model_response["choices"][0]["message"]["content"] = completion_response
            model_response["created"] = response.get("created", time.time())
            model_response["model"] = model
            model_response["usage"] = response.get("usage", 0)
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

            model_response = replicate.completion(
                model=model,
                messages=messages,
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

        elif model in litellm.anthropic_models:
            anthropic_key = (
                api_key or litellm.anthropic_key or os.environ.get("ANTHROPIC_API_KEY") or litellm.api_key
            )
            model_response = anthropic.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding, # for calculating input/output tokens
                api_key=anthropic_key,
                logging_obj=logging, 
            )
            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(model_response, model, custom_llm_provider="anthropic", logging_obj=logging)
                return response
            response = model_response
        elif model in litellm.nlp_cloud_models or custom_llm_provider == "nlp_cloud":
            nlp_cloud_key = (
                api_key or litellm.nlp_cloud_key or get_secret("NLP_CLOUD_API_KEY") or litellm.api_key
            )

            model_response = nlp_cloud.completion(
                model=model,
                messages=messages,
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
        elif model in litellm.aleph_alpha_models:
            aleph_alpha_key = (
                api_key or litellm.aleph_alpha_key or get_secret("ALEPH_ALPHA_API_KEY") or get_secret("ALEPHALPHA_API_KEY") or litellm.api_key
            )

            model_response = aleph_alpha.completion(
                model=model,
                messages=messages,
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
        elif model in litellm.openrouter_models or custom_llm_provider == "openrouter":
            openai.api_type = "openai"
            # not sure if this will work after someone first uses another API
            openai.api_base = (
                litellm.api_base
                if litellm.api_base is not None
                else "https://openrouter.ai/api/v1"
            )
            openai.api_version = None
            if litellm.organization:
                openai.organization = litellm.organization
            if api_key:
                openai.api_key = api_key
            elif litellm.openrouter_key:
                openai.api_key = litellm.openrouter_key
            else:
                openai.api_key = get_secret("OPENROUTER_API_KEY") or get_secret(
                    "OR_API_KEY"
                ) or litellm.api_key
            ## LOGGING
            logging.pre_call(input=messages, api_key=openai.api_key)
            ## COMPLETION CALL
            if litellm.headers:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    headers=litellm.headers,
                    **optional_params,
                )
            else:
                openrouter_site_url = get_secret("OR_SITE_URL")
                openrouter_app_name = get_secret("OR_APP_NAME")
                # if openrouter_site_url is None, set it to https://litellm.ai
                if openrouter_site_url is None:
                    openrouter_site_url = "https://litellm.ai"
                # if openrouter_app_name is None, set it to liteLLM
                if openrouter_app_name is None:
                    openrouter_app_name = "liteLLM"
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    headers={
                        "HTTP-Referer": openrouter_site_url,  # To identify your site
                        "X-Title": openrouter_app_name,  # To identify your app
                    },
                    **optional_params,
                )
            ## LOGGING
            logging.post_call(
                input=messages, api_key=openai.api_key, original_response=response
            )
        elif model in litellm.cohere_models:
            cohere_key = (
                api_key
                or litellm.cohere_key
                or get_secret("COHERE_API_KEY")
                or get_secret("CO_API_KEY")
                or litellm.api_key
            )
            model_response = cohere.completion(
                model=model,
                messages=messages,
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
        elif custom_llm_provider == "deepinfra": # for know this NEEDS to be above Hugging Face otherwise all calls to meta-llama/Llama-2-70b-chat-hf go to hf, we need this to go to deep infra if user sets provider to deep infra 
            # this can be called with the openai python package
            api_key = (
                api_key or
                litellm.api_key or
                litellm.openai_key or
                get_secret("DEEPINFRA_API_KEY")
            )
            ## LOGGING
            logging.pre_call(
                input=messages,
                api_key=api_key,
            )
            ## COMPLETION CALL
            openai.api_key = api_key # set key for deep infra 
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    api_base="https://api.deepinfra.com/v1/openai", # use the deepinfra api base
                    api_type="openai",
                    api_version=api_version, # default None
                    **optional_params,
                )
            except Exception as e:
                ## LOGGING - log the original exception returned
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=str(e),
                )
                raise e
            if "stream" in optional_params and optional_params["stream"] == True:
                response = CustomStreamWrapper(response, model, custom_llm_provider="openai", logging_obj=logging)
                return response
            ## LOGGING
            logging.post_call(
                input=messages,
                api_key=api_key,
                original_response=response,
                additional_args={"headers": litellm.headers},
            )
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
            model_response = huggingface_restapi.completion(
                model=model,
                messages=messages,
                api_base=api_base, # type: ignore
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding, 
                api_key=huggingface_key, 
                logging_obj=logging,
                custom_prompt_dict=litellm.custom_prompt_dict
            )
            if "stream" in optional_params and optional_params["stream"] == True:
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
        elif custom_llm_provider == "together_ai" or ("togethercomputer" in model) or (model  in litellm.together_ai_models):
            custom_llm_provider = "together_ai"
            together_ai_key = (
                api_key
                or litellm.togetherai_api_key
                or get_secret("TOGETHER_AI_TOKEN")
                or get_secret("TOGETHERAI_API_KEY")
                or litellm.api_key
            )
            
            model_response = together_ai.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=together_ai_key,
                logging_obj=logging
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

            # palm does not support streaming as yet :(
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
        elif model in litellm.ai21_models:
            custom_llm_provider = "ai21"
            ai21_key = (
                api_key
                or litellm.ai21_key
                or os.environ.get("AI21_API_KEY")
                or litellm.api_key
            )            
            model_response = ai21.completion(
                model=model,
                messages=messages,
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
            model_response = bedrock.completion(
                model=model,
                messages=messages,
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
                "http://localhost:11434"
                
            )
            if model in litellm.custom_prompt_dict:
                # check if the model has a registered custom prompt
                model_prompt_details = litellm.custom_prompt_dict[model]
                prompt = custom_prompt(
                    role_dict=model_prompt_details["roles"], 
                    initial_prompt_value=model_prompt_details["initial_prompt_value"],  
                    final_prompt_value=model_prompt_details["final_prompt_value"], 
                    messages=messages
                )
            else:
                prompt = prompt_factory(model=model, messages=messages, custom_llm_provider=custom_llm_provider)

            ## LOGGING
            logging.pre_call(
                input=prompt, api_key=None, additional_args={"api_base": api_base, "custom_prompt_dict": litellm.custom_prompt_dict}
            )
            if kwargs.get('acompletion', False) == True:    
                if optional_params.get("stream", False) == True:
                # assume all ollama responses are streamed
                    async_generator = ollama.async_get_ollama_response_stream(api_base, model, prompt, optional_params)
                    return async_generator

            generator = ollama.get_ollama_response_stream(api_base, model, prompt, optional_params)
            if optional_params.get("stream", False) == True:
                # assume all ollama responses are streamed
                return generator
            else:
                response_string = ""
                for chunk in generator:
                    response_string+=chunk['choices'][0]['delta']['content']
            
            ## RESPONSE OBJECT
            model_response["choices"][0]["message"]["content"] = response_string
            model_response["created"] = time.time()
            model_response["model"] = "ollama/" + model
            prompt_tokens = len(encoding.encode(prompt))
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
            prompt = " ".join([message["content"] for message in messages])
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
    try:
        import tenacity
    except:
        raise Exception("tenacity import failed please run `pip install tenacity`")

    retryer = tenacity.Retrying(stop=tenacity.stop_after_attempt(3), reraise=True)
    return retryer(completion, *args, **kwargs)


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
            print(f"{GREEN}LiteLLM: Making request to model: {model}{RESET}")
            future = executor.submit(completion, *args, model=model, **kwargs)
            if future.result() is not None:
                responses.append(future.result())
                print(f"{GREEN}LiteLLM: Model {model} returned response{RESET}")
            else:
                print(f"{RED}LiteLLM: Model {model } did not return a response{RESET}")

    return responses

### EMBEDDING ENDPOINTS ####################
@client
@timeout(  # type: ignore
    60
)  ## set timeouts, in case calls hang (e.g. Azure) - default is 60s, override with `force_timeout`
def embedding(
    model, 
    input=[], 
    api_key=None,
    api_base=None,
    # Optional params
    azure=False, 
    force_timeout=60, 
    litellm_call_id=None, 
    litellm_logging_obj=None,
    logger_fn=None, 
    caching=False,
    custom_llm_provider=None,
):
    model, custom_llm_provider = get_llm_provider(model, custom_llm_provider)
    try:
        response = None
        logging = litellm_logging_obj
        logging.update_environment_variables(model=model, user="", optional_params={}, litellm_params={"force_timeout": force_timeout, "azure": azure, "litellm_call_id": litellm_call_id, "logger_fn": logger_fn})
        if azure == True or custom_llm_provider == "azure":
            # azure configs
            openai.api_type = get_secret("AZURE_API_TYPE") or "azure"
            openai.api_base = get_secret("AZURE_API_BASE")
            openai.api_version = get_secret("AZURE_API_VERSION")
            openai.api_key = get_secret("AZURE_API_KEY")
            ## LOGGING
            logging.pre_call(
                input=input,
                api_key=openai.api_key,
                additional_args={
                    "api_type": openai.api_type,
                    "api_base": openai.api_base,
                    "api_version": openai.api_version,
                },
            )
            ## EMBEDDING CALL
            response = openai.Embedding.create(input=input, engine=model)

            ## LOGGING
            logging.post_call(input=input, api_key=openai.api_key, original_response=response)
        elif model in litellm.open_ai_embedding_models:
            openai.api_type = "openai"
            openai.api_base = "https://api.openai.com/v1"
            openai.api_version = None
            openai.api_key = get_secret("OPENAI_API_KEY")
            ## LOGGING
            logging.pre_call(
                input=input,
                api_key=openai.api_key,
                additional_args={
                    "api_type": openai.api_type,
                    "api_base": openai.api_base,
                    "api_version": openai.api_version,
                },
            )
            ## EMBEDDING CALL
            response = openai.Embedding.create(input=input, model=model)

            ## LOGGING
            logging.post_call(input=input, api_key=openai.api_key, original_response=response)
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
            response = huggingface_restapi.embedding(
                model=model,
                input=input,
                encoding=encoding,
                api_key=api_key,
                api_base=api_base,
                logging_obj=logging,
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
def text_completion(*args, **kwargs):
    if "prompt" in kwargs:
        messages = [{"role": "system", "content": kwargs["prompt"]}]
        kwargs["messages"] = messages
        kwargs.pop("prompt")
        response = completion(*args, **kwargs) # assume the response is the openai response object 
        print(f"response: {response}")
        formatted_response_obj = {
            "id": response["id"],
            "object": "text_completion",
            "created": response["created"],
            "model": response["model"],
            "choices": [
            {
                "text": response["choices"][0]["message"]["content"],
                "index": response["choices"][0]["index"],
                "logprobs": None,
                "finish_reason": response["choices"][0]["finish_reason"]
            }
            ],
            "usage": response["usage"]
        }
        return formatted_response_obj
    else:
        raise ValueError("please pass prompt into the `text_completion` endpoint - `text_completion(model, prompt='hello world')`")

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
    openai.api_type = "open_ai"
    openai.api_version = None
    openai.api_base = "https://api.openai.com/v1"
    response = openai.Moderation.create(input)
    return response

####### HELPER FUNCTIONS ################
## Set verbose to true -> ```litellm.set_verbose = True```
def print_verbose(print_statement):
    if litellm.set_verbose:
        print(f"LiteLLM: {print_statement}")

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
        # "usage": {
        #     "prompt_tokens": 0,  # Modify as needed
        #     "completion_tokens": 0,  # Modify as needed
        #     "total_tokens": 0  # Modify as needed
        # }
    }

    # Extract the "content" strings from the nested dictionaries within "choices"
    content_list = []

    if "function_call" in chunks[0]["choices"][0]["delta"]:
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
                content_list.append(content)

        # Combine the "content" strings into a single string
        combined_content = "".join(content_list)

        # Update the "content" field within the response dictionary
        response["choices"][0]["message"]["content"] = combined_content


    # # Update usage information if needed
    # response["usage"]["completion_tokens"] = token

    return response
