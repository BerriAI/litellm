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
    mock_completion_streaming_obj
)
from .llms import anthropic
from .llms import together_ai
from .llms import ai21
from .llms import sagemaker
from .llms import bedrock
from .llms import huggingface_restapi
from .llms import replicate
from .llms import aleph_alpha
from .llms import nlp_cloud
from .llms import baseten
from .llms import vllm
from .llms import ollama
from .llms import cohere
import tiktoken
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Dict

encoding = tiktoken.get_encoding("cl100k_base")
from litellm.utils import (
    get_secret,
    CustomStreamWrapper,
    ModelResponse,
    read_config_args,
)

####### ENVIRONMENT VARIABLES ###################
dotenv.load_dotenv()  # Loading env variables using dotenv


####### COMPLETION ENDPOINTS ################
#############################################
async def acompletion(*args, **kwargs):
    loop = asyncio.get_event_loop()

    # Use a partial function to pass your keyword arguments
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

def mock_completion(model: str, messages: List, stream: bool = False, mock_response: str = "This is a mock request", **kwargs):
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
    temperature: float = 1,
    top_p: float = 1,
    n: int = 1,
    stream: bool = False,
    stop=None,
    max_tokens: float = float("inf"),
    presence_penalty: float = 0,
    frequency_penalty=0,
    logit_bias: dict = {},
    user: str = "",
    deployment_id = None,
    # Optional liteLLM function params
    *,
    return_async=False,
    mock_response: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
    api_base: Optional[str] = None,
    force_timeout=600,
    num_beams=1,
    logger_fn=None,
    verbose=False,
    azure=False,
    custom_llm_provider=None,
    litellm_call_id=None,
    litellm_logging_obj=None,
    use_client=False,
    id=None, # this is an optional param to tag individual completion calls 
    # model specific optional params
    top_k=40,# used by text-bison only
    task: Optional[str]="text-generation-inference", # used by huggingface inference endpoints
    return_full_text: bool = False, # used by huggingface TGI
    remove_input: bool = True, # used by nlp cloud models - prevents input text from being returned as part of output
    request_timeout=0,  # unused var for old version of OpenAI API
    fallbacks=[],
    caching = False,
    cache_params = {}, # optional to specify metadata for caching
) -> ModelResponse:
    # If `mock_response` is set, execute the `mock_completion` method instead.
    if mock_response:
        return mock_completion(model, messages, stream=stream, mock_response=mock_response)
    
    args = locals()
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
        if deployment_id != None: # azure llms
                model=deployment_id
                custom_llm_provider="azure"
        elif (
            model.split("/", 1)[0] in litellm.provider_list
        ):  # allow custom provider to be passed in via the model name "azure/chatgpt-test"
            custom_llm_provider = model.split("/", 1)[0]
            model = model.split("/", 1)[1]
        model, custom_llm_provider = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
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
            deployment_id=deployment_id,
            # params to identify the model
            model=model,
            custom_llm_provider=custom_llm_provider,
            top_k=top_k,
            task=task,
            remove_input=remove_input,
            return_full_text=return_full_text
        )
        # For logging - save the values of the litellm-specific params passed in
        litellm_params = get_litellm_params(
            return_async=return_async,
            api_key=api_key,
            force_timeout=force_timeout,
            logger_fn=logger_fn,
            verbose=verbose,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            litellm_call_id=litellm_call_id,
            model_alias_map=litellm.model_alias_map,
            completion_call_id=id
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
                response = CustomStreamWrapper(response, model, logging_obj=logging)
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
                response = CustomStreamWrapper(response, model, logging_obj=logging)
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
                response = CustomStreamWrapper(response, model, custom_llm_provider="openai", logging_obj=logging)
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
                get_secret("REPLICATE_API_KEY")
                or get_secret("REPLICATE_API_TOKEN")
                or api_key
                or litellm.replicate_key
                or litellm.api_key
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
                response = CustomStreamWrapper(model_response, model, logging_obj=logging)
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
                response = CustomStreamWrapper(model_response, model, logging_obj=logging)
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
                response = CustomStreamWrapper(model_response, model, logging_obj=logging)
                return response
            response = model_response
        elif (
            (
                model in litellm.huggingface_models and 
                custom_llm_provider!="custom" # if users use a hf model, with a custom/provider. See implementation of custom_llm_provider == custom
            ) or 
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
        elif model in litellm.vertex_chat_models or model in litellm.vertex_code_chat_models:
            try:
                import vertexai
            except:
                raise Exception("vertexai import failed please run `pip install google-cloud-aiplatform`")
            from vertexai.preview.language_models import ChatModel, CodeChatModel, InputOutputTextPair

            vertexai.init(
                project=litellm.vertex_project, location=litellm.vertex_location
            )
            # vertexai does not use an API key, it looks for credentials.json in the environment

            prompt = " ".join([message["content"] for message in messages])
            ## LOGGING
            logging.pre_call(input=prompt, api_key=None)
            if model in litellm.vertex_chat_models:
                chat_model = ChatModel.from_pretrained(model)
            else: # vertex_code_chat_models
                chat_model = CodeChatModel.from_pretrained(model)

            chat = chat_model.start_chat()

            if stream:
                model_response = chat.send_message_streaming(prompt, **optional_params)
                response = CustomStreamWrapper(
                    model_response, model, custom_llm_provider="vertexai", logging_obj=logging
                )
                return response

            completion_response = chat.send_message(prompt, **optional_params)

            ## LOGGING
            logging.post_call(
                input=prompt, api_key=None, original_response=completion_response
            )

            ## RESPONSE OBJECT
            model_response["choices"][0]["message"]["content"] = str(completion_response)
            model_response["created"] = time.time()
            model_response["model"] = model
            response = model_response
        elif model in litellm.vertex_text_models or model in litellm.vertex_code_text_models:
            try:
                import vertexai
            except:
                raise Exception("vertexai import failed please run `pip install google-cloud-aiplatform`")
            from vertexai.language_models import TextGenerationModel, CodeGenerationModel

            vertexai.init(
                project=litellm.vertex_project, location=litellm.vertex_location
            )
            # vertexai does not use an API key, it looks for credentials.json in the environment

            prompt = " ".join([message["content"] for message in messages])
            ## LOGGING
            logging.pre_call(input=prompt, api_key=None)
            
            if model in litellm.vertex_text_models:
                vertex_model = TextGenerationModel.from_pretrained(model)
            else:
                vertex_model = CodeGenerationModel.from_pretrained(model)

            if stream:
                model_response = vertex_model.predict_streaming(prompt, **optional_params)
                response = CustomStreamWrapper(
                    model_response, model, custom_llm_provider="vertexai", logging_obj=logging
                )
                return response

            completion_response = vertex_model.predict(prompt, **optional_params)

            ## LOGGING
            logging.post_call(
                input=prompt, api_key=None, original_response=completion_response
            )
            ## RESPONSE OBJECT
            model_response["choices"][0]["message"]["content"] = str(completion_response)
            model_response["created"] = time.time()
            model_response["model"] = model
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

            if "stream" in optional_params and optional_params["stream"] == True: ## [BETA]
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    iter(model_response), model, custom_llm_provider="sagemaker", logging_obj=logging
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
                stream=stream,
            )


            if stream == True:
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
            endpoint = (
                litellm.api_base if litellm.api_base is not None else api_base
            )
            prompt = " ".join([message["content"] for message in messages])

            ## LOGGING
            logging.pre_call(
                input=prompt, api_key=None, additional_args={"endpoint": endpoint}
            )
            generator = ollama.get_ollama_response_stream(endpoint, model, prompt)
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
                    'top_k': top_k,
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
            model=model, custom_llm_provider=custom_llm_provider, original_exception=e
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
    temperature: float = 1,
    top_p: float = 1,
    n: int = 1,
    stream: bool = False,
    stop=None,
    max_tokens: float = float("inf"),
    presence_penalty: float = 0,
    frequency_penalty=0,
    logit_bias: dict = {},
    user: str = "",
    # Optional liteLLM function params
    *,
    return_async=False,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
    api_base: Optional[str] = None,
    force_timeout=600,
    # used by text-bison only
    top_k=40,
    custom_llm_provider=None,):
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
            custom_llm_provider=custom_llm_provider,
            top_k=top_k,
        )
        results = vllm.batch_completions(model=model, messages=batch_messages, custom_prompt_dict=litellm.custom_prompt_dict, optional_params=optional_params)
    else:
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        with ThreadPoolExecutor(max_workers=100) as executor:
            for sub_batch in chunks(batch_messages, 100):
                for message_list in sub_batch:
                    kwargs_modified = args
                    kwargs_modified["messages"] = message_list
                    future = executor.submit(completion, **kwargs_modified)
                    completions.append(future)

        # Retrieve the results from the futures
        results = [future.result() for future in completions]
    return results

### EMBEDDING ENDPOINTS ####################
@client
@timeout(  # type: ignore
    60
)  ## set timeouts, in case calls hang (e.g. Azure) - default is 60s, override with `force_timeout`
def embedding(
    model, input=[], azure=False, force_timeout=60, litellm_call_id=None, litellm_logging_obj=None, logger_fn=None, caching=False,
):
    try:
        response = None
        logging = litellm_logging_obj
        logging.update_environment_variables(model=model, user="", optional_params={}, litellm_params={"force_timeout": force_timeout, "azure": azure, "litellm_call_id": litellm_call_id, "logger_fn": logger_fn})
        if azure == True:
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
        else:
            args = locals()
            raise ValueError(f"No valid embedding model args passed in - {args}")
        ## LOGGING
        logging.post_call(input=input, api_key=openai.api_key, original_response=response)
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
        return completion(*args, **kwargs)


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
