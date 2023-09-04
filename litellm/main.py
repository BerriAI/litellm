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
    install_and_import,
    CustomStreamWrapper,
    read_config_args,
    completion_with_fallbacks,
)
from .llms import anthropic
from .llms import together_ai
from .llms import ai21
from .llms import sagemaker
from .llms import bedrock
from .llms.huggingface_restapi import HuggingfaceRestAPILLM
from .llms.baseten import BasetenLLM
from .llms.aleph_alpha import AlephAlphaLLM
import tiktoken
from concurrent.futures import ThreadPoolExecutor

encoding = tiktoken.get_encoding("cl100k_base")
from litellm.utils import (
    get_secret,
    install_and_import,
    CustomStreamWrapper,
    ModelResponse,
    read_config_args,
)
from litellm.utils import (
    get_ollama_response_stream,
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
    return await loop.run_in_executor(None, func_with_context)


@client
@timeout(  # type: ignore
    600
)  ## set timeouts, in case calls hang (e.g. Azure) - default is 600s, override with `force_timeout`
def completion(
    model,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages=[],
    functions=[],
    function_call="",  # optional params
    temperature=1,
    top_p=1,
    n=1,
    stream=False,
    stop=None,
    max_tokens=float("inf"),
    presence_penalty=0,
    frequency_penalty=0,
    logit_bias={},
    user="",
    deployment_id=None,
    # Optional liteLLM function params
    *,
    return_async=False,
    api_key=None,
    api_version=None,
    force_timeout=600,
    num_beams=1,
    logger_fn=None,
    verbose=False,
    azure=False,
    custom_llm_provider=None,
    api_base=None,
    litellm_call_id=None,
    litellm_logging_obj=None,
    use_client=False,
    id=None, # this is an optional param to tag individual completion calls 
    # model specific optional params
    # used by text-bison only
    top_k=40,
    request_timeout=0,  # unused var for old version of OpenAI API
    fallbacks=[],
) -> ModelResponse:
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
        if azure:  # this flag is deprecated, remove once notebooks are also updated.
            custom_llm_provider = "azure"
        elif (
            model.split("/", 1)[0] in litellm.provider_list
        ):  # allow custom provider to be passed in via the model name "azure/chatgpt-test"
            custom_llm_provider = model.split("/", 1)[0]
            model = model.split("/", 1)[1]
            if (
                "replicate" == custom_llm_provider and "/" not in model
            ):  # handle the "replicate/llama2..." edge-case
                model = custom_llm_provider + "/" + model
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
        logging.update_environment_variables(model=model, optional_params=optional_params, litellm_params=litellm_params)
        if custom_llm_provider == "azure":
            # azure configs
            openai.api_type = "azure"

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("AZURE_API_BASE")
            )

            openai.api_version = (
                litellm.api_version
                if litellm.api_version is not None
                else get_secret("AZURE_API_VERSION")
            )
            if not api_key and litellm.azure_key:
                api_key = litellm.azure_key
            elif not api_key and get_secret("AZURE_API_KEY"):
                api_key = get_secret("AZURE_API_KEY")

            ## LOGGING
            logging.pre_call(
                input=messages,
                api_key=openai.api_key,
                additional_args={
                    "headers": litellm.headers,
                    "api_version": openai.api_version,
                    "api_base": openai.api_base,
                },
            )
            ## COMPLETION CALL
            response = openai.ChatCompletion.create(
                engine=model,
                messages=messages,
                headers=litellm.headers,
                api_key=api_key,
                api_base=api_base,
                **optional_params,
            )
            if "stream" in optional_params and optional_params["stream"] == True:
                response = CustomStreamWrapper(response, model, logging_obj=logging)
                return response
            ## LOGGING
            logging.post_call(
                input=messages,
                api_key=openai.api_key,
                original_response=response,
                additional_args={
                    "headers": litellm.headers,
                    "api_version": openai.api_version,
                    "api_base": openai.api_base,
                },
            )
        elif (
            model in litellm.open_ai_chat_completion_models
            or custom_llm_provider == "custom_openai"
            or "ft:gpt-3.5-turbo" in model  # finetuned gpt-3.5-turbo
        ):  # allow user to make an openai call with a custom base
            openai.api_type = "openai"
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
            if not api_key and litellm.openai_key:
                api_key = litellm.openai_key
            elif not api_key and get_secret("OPENAI_API_KEY"):
                api_key = get_secret("OPENAI_API_KEY")

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
        ):
            openai.api_type = "openai"
            openai.api_base = (
                litellm.api_base
                if litellm.api_base is not None
                else "https://api.openai.com/v1"
            )
            openai.api_version = None
            # set API KEY
            if not api_key and litellm.openai_key:
                api_key = litellm.openai_key
            elif not api_key and get_secret("OPENAI_API_KEY"):
                api_key = get_secret("OPENAI_API_KEY")

            openai.api_key = api_key

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
                    "api_base": openai.api_base,
                    "api_type": openai.api_type,
                },
            )
            ## COMPLETION CALL
            if litellm.headers:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    headers=litellm.headers,
                )
            else:
                response = openai.Completion.create(model=model, prompt=prompt, **optional_params)
            
            if "stream" in optional_params and optional_params["stream"] == True:
                response = CustomStreamWrapper(response, model, logging_obj=logging)
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
            model_response["created"] = response["created"]
            model_response["model"] = model
            model_response["usage"] = response["usage"]
            response = model_response
        elif "replicate" in model or custom_llm_provider == "replicate":
            # import replicate/if it fails then pip install replicate
            install_and_import("replicate")
            import replicate

            # Setting the relevant API KEY for replicate, replicate defaults to using os.environ.get("REPLICATE_API_TOKEN")
            replicate_key = os.environ.get("REPLICATE_API_TOKEN")
            if replicate_key == None:
                # user did not set REPLICATE_API_TOKEN in .env
                replicate_key = (
                    get_secret("REPLICATE_API_KEY")
                    or get_secret("REPLICATE_API_TOKEN")
                    or api_key
                    or litellm.replicate_key
                )
                # set replicate key
                os.environ["REPLICATE_API_TOKEN"] = str(replicate_key)
            prompt = " ".join([message["content"] for message in messages])
            input = {"prompt": prompt}
            if "max_tokens" in optional_params:
                input["max_length"] = max_tokens  # for t5 models
                input["max_new_tokens"] = max_tokens  # for llama2 models
            ## LOGGING
            logging.pre_call(
                input=prompt,
                api_key=replicate_key,
                additional_args={
                    "complete_input_dict": input,
                    "max_tokens": max_tokens,
                },
            )
            ## COMPLETION CALL
            output = replicate.run(model, input=input)
            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                # let the stream handler know this is replicate
                response = CustomStreamWrapper(output, "replicate", logging_obj=logging)
                return response
            response = ""
            for item in output:
                response += item
            completion_response = response
            ## LOGGING
            logging.post_call(
                input=prompt,
                api_key=replicate_key,
                original_response=completion_response,
                additional_args={
                    "complete_input_dict": input,
                    "max_tokens": max_tokens,
                },
            )
            ## USAGE
            prompt_tokens = len(encoding.encode(prompt))
            completion_tokens = len(encoding.encode(completion_response))
            ## RESPONSE OBJECT
            model_response["choices"][0]["message"]["content"] = completion_response
            model_response["created"] = time.time()
            model_response["model"] = model
            model_response["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            response = model_response
        elif model in litellm.anthropic_models:
            anthropic_key = (
                api_key or litellm.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
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
        elif model in litellm.aleph_alpha_models:
            aleph_alpha_key = (
                api_key or litellm.aleph_alpha_key or os.environ.get("ALEPH_ALPHA_API_KEY")
            )

            aleph_alpha_client = AlephAlphaLLM(
                encoding=encoding,
                default_max_tokens_to_sample=litellm.max_tokens,
                api_key=aleph_alpha_key,
                logging_obj=logging # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
            )

            model_response = aleph_alpha_client.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
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
                )
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
            # import cohere/if it fails then pip install cohere
            install_and_import("cohere")
            import cohere

            cohere_key = (
                api_key
                or litellm.cohere_key
                or get_secret("COHERE_API_KEY")
                or get_secret("CO_API_KEY")
            )
            co = cohere.Client(cohere_key)
            prompt = " ".join([message["content"] for message in messages])
            ## LOGGING
            logging.pre_call(input=prompt, api_key=cohere_key)
            ## COMPLETION CALL
            response = co.generate(model=model, prompt=prompt, **optional_params)
            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(response, model, logging_obj=logging)
                return response
            ## LOGGING
            logging.post_call(
                input=prompt, api_key=cohere_key, original_response=response
            )
            ## USAGE
            completion_response = response[0].text
            prompt_tokens = len(encoding.encode(prompt))
            completion_tokens = len(encoding.encode(completion_response))
            ## RESPONSE OBJECT
            model_response["choices"][0]["message"]["content"] = completion_response
            model_response["created"] = time.time()
            model_response["model"] = model
            model_response["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            response = model_response
        elif (
            model in litellm.huggingface_models or custom_llm_provider == "huggingface"
        ):
            custom_llm_provider = "huggingface"
            huggingface_key = (
                api_key
                or litellm.huggingface_key
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_API_KEY")
            )
            huggingface_client = HuggingfaceRestAPILLM(
                encoding=encoding, api_key=huggingface_key, logging_obj=logging
            )
            model_response = huggingface_client.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
            )
            if "stream" in optional_params and optional_params["stream"] == True:
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response, model, custom_llm_provider="huggingface", logging_obj=logging
                )
                return response
            response = model_response
        elif custom_llm_provider == "together_ai" or ("togethercomputer" in model):
            custom_llm_provider = "together_ai"
            together_ai_key = (
                api_key
                or litellm.togetherai_api_key
                or get_secret("TOGETHER_AI_TOKEN")
                or get_secret("TOGETHERAI_API_KEY")
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
        elif model in litellm.vertex_chat_models:
            import vertexai
            from vertexai.preview.language_models import ChatModel, InputOutputTextPair

            vertexai.init(
                project=litellm.vertex_project, location=litellm.vertex_location
            )
            # vertexai does not use an API key, it looks for credentials.json in the environment

            prompt = " ".join([message["content"] for message in messages])
            ## LOGGING
            logging.pre_call(input=prompt, api_key=None)

            chat_model = ChatModel.from_pretrained(model)

            chat = chat_model.start_chat()
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
        elif model in litellm.vertex_text_models:
            import vertexai
            from vertexai.language_models import TextGenerationModel

            vertexai.init(
                project=litellm.vertex_project, location=litellm.vertex_location
            )
            # vertexai does not use an API key, it looks for credentials.json in the environment

            prompt = " ".join([message["content"] for message in messages])
            ## LOGGING
            logging.pre_call(input=prompt, api_key=None)

            vertex_model = TextGenerationModel.from_pretrained(model)
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
            
            # TODO: Add streaming for sagemaker
            # if "stream" in optional_params and optional_params["stream"] == True:
            #     # don't try to access stream object,
            #     response = CustomStreamWrapper(
            #         model_response, model, custom_llm_provider="ai21", logging_obj=logging
            #     )
            #     return response
            
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
                logging_obj=logging
            )
            
            # TODO: Add streaming for bedrock
            # if "stream" in optional_params and optional_params["stream"] == True:
            #     # don't try to access stream object,
            #     response = CustomStreamWrapper(
            #         model_response, model, custom_llm_provider="ai21", logging_obj=logging
            #     )
            #     return response
            
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

            generator = get_ollama_response_stream(endpoint, model, prompt)
            # assume all responses are streamed
            return generator
        elif (
            custom_llm_provider == "baseten"
            or litellm.api_base == "https://app.baseten.co"
        ):
            custom_llm_provider = "baseten"
            baseten_key = (
                api_key or litellm.baseten_key or os.environ.get("BASETEN_API_KEY")
            )
            baseten_client = BasetenLLM(
                encoding=encoding, api_key=baseten_key, logging_obj=logging
            )
            model_response = baseten_client.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
            )
            if inspect.isgenerator(model_response) or ("stream" in optional_params and optional_params["stream"] == True):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response, model, custom_llm_provider="baseten", logging_obj=logging
                )
                return response
            response = model_response
        elif custom_llm_provider == "petals" or (
            litellm.api_base and "chat.petals.dev" in litellm.api_base
        ):
            url = "https://chat.petals.dev/api/v1/generate"
            import requests

            prompt = " ".join([message["content"] for message in messages])

            ## LOGGING
            logging.pre_call(
                input=prompt,
                api_key=None,
                additional_args={"url": url, "max_new_tokens": 100},
            )

            response = requests.post(
                url, data={"inputs": prompt, "max_new_tokens": 100, "model": model}
            )
            ## LOGGING
            logging.post_call(
                input=prompt,
                api_key=None,
                original_response=response.text,
                additional_args={"url": url, "max_new_tokens": 100},
            )

            completion_response = response.json()["outputs"]

            # RESPONSE OBJECT
            model_response["choices"][0]["message"]["content"] = completion_response
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
    import tenacity

    retryer = tenacity.Retrying(stop=tenacity.stop_after_attempt(3), reraise=True)
    return retryer(completion, *args, **kwargs)


def batch_completion(*args, **kwargs):
    batch_messages = args[1] if len(args) > 1 else kwargs.get("messages")
    completions = []
    with ThreadPoolExecutor() as executor:
        for message_list in batch_messages:
            if len(args) > 1:
                args_modified = list(args)
                args_modified[1] = message_list
                future = executor.submit(completion, *args_modified)
            else:
                kwargs_modified = dict(kwargs)
                kwargs_modified["messages"] = message_list
                future = executor.submit(completion, *args, **kwargs_modified)
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
    model, input=[], azure=False, force_timeout=60, litellm_call_id=None, litellm_logging_obj=None, logger_fn=None
):
    try:
        response = None
        logging = litellm_logging_obj
        logging.update_environment_variables(model=model, optional_params={}, litellm_params={"force_timeout": force_timeout, "azure": azure, "litellm_call_id": litellm_call_id, "logger_fn": logger_fn})
        if azure == True:
            # azure configs
            openai.api_type = "azure"
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
        if random.random() <= 0.3:
            print("Get help - https://discord.com/invite/wuPM9dRgDw")


def config_completion(**kwargs):
    if litellm.config_path != None:
        config_args = read_config_args(litellm.config_path)
        # overwrite any args passed in with config args
        return completion(**kwargs, **config_args)
    else:
        raise ValueError(
            "No config path set, please set a config path using `litellm.config_path = 'path/to/config.json'`"
        )
