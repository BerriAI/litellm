import time
from typing import Final

import httpx

import litellm
from litellm.exceptions import LiteLLMUnknownProvider
from litellm.llms.custom_llm import CustomLLM, custom_chat_llm_router
from litellm.native.chat_providers.support import get_encoding, print_verbose
from litellm.types.completion import CompletionDispatchContext, CompletionDispatchResult
from litellm.utils import CustomStreamWrapper


def _custom_api_first_output(resp: httpx.Response | None) -> str:
    return resp.json()["data"][0]["output"][0]


def complete_custom(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    api_base: Final = ctx.api_base
    headers: Final = ctx.headers
    kwargs: Final = ctx.kwargs
    max_tokens: Final = ctx.max_tokens
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    temperature: Final = ctx.temperature
    top_p: Final = ctx.top_p

    url: Final = litellm.api_base or api_base or ""
    if url is None or url == "":
        raise ValueError("api_base not set. Set api_base or litellm.api_base for custom endpoints")

    """
    assume input to custom LLM api bases follow this format:
    resp = litellm.module_level_client.post(
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
    prompt: Final = " ".join([message["content"] for message in messages])
    resp: Final = litellm.module_level_client.post(
        url,
        headers=headers,
        json={
            "model": model,
            "params": {
                "prompt": [prompt],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": kwargs.get("top_k"),
            },
            **kwargs.get("extra_body", {}),
        },
    )
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
    string_response: Final = _custom_api_first_output(resp)
    ## RESPONSE OBJECT
    model_response.choices[0].message.content = string_response
    model_response.created = int(time.time())
    model_response.model = model
    return model_response


def complete_custom_providers(
    ctx: CompletionDispatchContext,
) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base: Final = ctx.api_base
    api_key: Final = ctx.api_key
    client: Final = ctx.client
    custom_llm_provider: Final = ctx.custom_llm_provider
    custom_prompt_dict: Final = ctx.custom_prompt_dict
    headers = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    custom_handler: CustomLLM | None = None
    for item in litellm.custom_provider_map:
        if item["provider"] == custom_llm_provider:
            custom_handler = item["custom_handler"]

    if custom_handler is None:
        raise LiteLLMUnknownProvider(model=model, custom_llm_provider=custom_llm_provider)

    ## ROUTE LLM CALL ##
    handler_fn: Final = custom_chat_llm_router(async_fn=acompletion, stream=stream, custom_llm=custom_handler)

    headers = headers or litellm.headers or {}

    ## CALL FUNCTION
    response: Final = handler_fn(
        model=model,
        messages=messages,
        headers=headers,
        model_response=model_response,
        print_verbose=print_verbose,
        api_key=api_key,
        api_base=api_base,
        acompletion=acompletion,
        logging_obj=logging,
        optional_params=optional_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        timeout=timeout,
        custom_prompt_dict=custom_prompt_dict,
        client=client,  # pass AsyncOpenAI, OpenAI client
        encoding=get_encoding(),
    )
    if stream is True:
        return CustomStreamWrapper(
            completion_stream=response,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging,
        )

    return response  # pyright: ignore[reportReturnType]  # provider SDK return type is broader than the dispatch contract
