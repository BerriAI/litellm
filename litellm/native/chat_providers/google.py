from typing import Final

import litellm
from litellm.litellm_core_utils.core_helpers import safe_deep_copy
from litellm.llms.gemini.common_utils import get_api_key_from_env
from litellm.llms.vertex_ai import vertex_ai_non_gemini
from litellm.llms.vertex_ai.common_utils import VertexAIModelRoute, get_vertex_ai_model_route
from litellm.native.chat_providers.handlers import (
    base_llm_http_handler,
    gdc_transformation,
    vertex_chat_completion,
    vertex_gemma_chat_completion,
    vertex_model_garden_chat_completion,
    vertex_partner_models_chat_completion,
)
from litellm.native.chat_providers.support import dispatch_client_http, get_encoding, print_verbose
from litellm.secret_managers.main import get_secret_str
from litellm.types.completion import CompletionDispatchContext, CompletionDispatchResult
from litellm.utils import CustomStreamWrapper, get_secret


def complete_vertex_ai_beta(
    ctx: CompletionDispatchContext,
) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key: Final = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    timeout: Final = ctx.timeout

    vertex_ai_project: Final = (
        optional_params.pop("vertex_project", None)
        or optional_params.pop("vertex_ai_project", None)
        or litellm.vertex_project
        or get_secret("VERTEXAI_PROJECT")
    )
    vertex_ai_location: Final = (
        optional_params.pop("vertex_location", None)
        or optional_params.pop("vertex_ai_location", None)
        or litellm.vertex_location
        or get_secret("VERTEXAI_LOCATION")
    )
    vertex_credentials: Final = (
        optional_params.pop("vertex_credentials", None)
        or optional_params.pop("vertex_ai_credentials", None)
        or get_secret("VERTEXAI_CREDENTIALS")
    )

    gemini_api_key: Final = (
        api_key
        or get_api_key_from_env()
        or get_secret("PALM_API_KEY")  # older palm api key should also work
        or litellm.api_key
    )

    api_base = api_base or litellm.api_base or get_secret("GEMINI_API_BASE")
    new_params: Final = safe_deep_copy(optional_params or {})
    return vertex_chat_completion.completion(
        model=model,
        messages=messages,
        model_response=model_response,
        print_verbose=print_verbose,
        optional_params=new_params,
        litellm_params=litellm_params,
        logger_fn=logger_fn,
        encoding=get_encoding(),
        vertex_location=vertex_ai_location,
        vertex_project=vertex_ai_project,
        vertex_credentials=vertex_credentials,
        gemini_api_key=gemini_api_key,
        logging_obj=logging,
        acompletion=acompletion,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        client=client,
        api_base=api_base,
        extra_headers=headers,
    )


def complete_vertex_ai(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    client: Final = dispatch_client_http(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    custom_prompt_dict: Final = ctx.custom_prompt_dict
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logger_fn: Final = ctx.logger_fn
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response = ctx.model_response
    optional_params: Final = ctx.optional_params
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    vertex_ai_project: Final = (
        optional_params.pop("vertex_project", None)
        or optional_params.pop("vertex_ai_project", None)
        or litellm.vertex_project
        or get_secret("VERTEXAI_PROJECT")
    )
    vertex_ai_location: Final = (
        optional_params.pop("vertex_location", None)
        or optional_params.pop("vertex_ai_location", None)
        or litellm.vertex_location
        or get_secret("VERTEXAI_LOCATION")
    )
    vertex_credentials: Final = (
        optional_params.pop("vertex_credentials", None)
        or optional_params.pop("vertex_ai_credentials", None)
        or get_secret("VERTEXAI_CREDENTIALS")
    )

    api_base = api_base or litellm.api_base or get_secret("VERTEXAI_API_BASE")

    new_params: Final = safe_deep_copy(optional_params or {})
    model_route: Final = get_vertex_ai_model_route(model=model, litellm_params=litellm_params)

    if model_route == VertexAIModelRoute.PARTNER_MODELS:
        model_response = vertex_partner_models_chat_completion.completion(
            model=model,
            messages=messages,
            model_response=model_response,
            print_verbose=print_verbose,
            optional_params=new_params,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            encoding=get_encoding(),
            api_base=api_base,
            vertex_location=vertex_ai_location,
            vertex_project=vertex_ai_project,
            vertex_credentials=vertex_credentials,
            logging_obj=logging,
            acompletion=acompletion,
            headers=headers,
            custom_prompt_dict=custom_prompt_dict,
            timeout=timeout,
            client=client,
        )
    elif model_route == VertexAIModelRoute.GEMINI:
        model_response = vertex_chat_completion.completion(
            model=model,
            messages=messages,
            model_response=model_response,
            print_verbose=print_verbose,
            optional_params=new_params,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            encoding=get_encoding(),
            vertex_location=vertex_ai_location,
            vertex_project=vertex_ai_project,
            vertex_credentials=vertex_credentials,
            gemini_api_key=None,
            logging_obj=logging,
            acompletion=acompletion,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            client=client,
            api_base=api_base,
            extra_headers=headers,
        )
    elif model_route == VertexAIModelRoute.GEMMA:
        # Vertex Gemma Models with custom prediction endpoint
        model_response = vertex_gemma_chat_completion.completion(
            model=model,
            messages=messages,
            model_response=model_response,
            print_verbose=print_verbose,
            optional_params=new_params,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            encoding=get_encoding(),
            api_base=api_base,
            vertex_location=vertex_ai_location,
            vertex_project=vertex_ai_project,
            vertex_credentials=vertex_credentials,
            logging_obj=logging,
            acompletion=acompletion,
            headers=headers,
            custom_prompt_dict=custom_prompt_dict,
            timeout=timeout,
            client=client,
        )
    elif model_route == VertexAIModelRoute.MODEL_GARDEN:
        # Vertex Model Garden - OpenAI compatible models
        model_response = vertex_model_garden_chat_completion.completion(
            model=model,
            messages=messages,
            model_response=model_response,
            print_verbose=print_verbose,
            optional_params=new_params,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            encoding=get_encoding(),
            api_base=api_base,
            vertex_location=vertex_ai_location,
            vertex_project=vertex_ai_project,
            vertex_credentials=vertex_credentials,
            logging_obj=logging,
            acompletion=acompletion,
            headers=headers,
            custom_prompt_dict=custom_prompt_dict,
            timeout=timeout,
            client=client,
        )
    elif model_route == VertexAIModelRoute.AGENT_ENGINE:
        # Vertex AI Agent Engine (Reasoning Engines)
        from litellm.llms.vertex_ai.agent_engine.transformation import (
            VertexAgentEngineConfig,
        )

        vertex_agent_engine_config: Final = VertexAgentEngineConfig()

        # Update litellm_params with vertex credentials
        litellm_params["vertex_project"] = vertex_ai_project
        litellm_params["vertex_location"] = vertex_ai_location
        litellm_params["vertex_credentials"] = vertex_credentials

        model_response = base_llm_http_handler.completion(
            model=model,
            stream=stream,
            messages=messages,
            model_response=model_response,
            optional_params=new_params,
            litellm_params=litellm_params,
            encoding=get_encoding(),
            api_key=None,
            api_base=api_base,
            logging_obj=logging,
            acompletion=acompletion,
            timeout=timeout,
            client=client,
            custom_llm_provider="vertex_ai",
            provider_config=vertex_agent_engine_config,
            headers=headers or {},
        )
    else:  # VertexAIModelRoute.NON_GEMINI
        model_response = vertex_ai_non_gemini.completion(
            model=model,
            messages=messages,
            model_response=model_response,
            print_verbose=print_verbose,
            optional_params=new_params,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            encoding=get_encoding(),
            vertex_location=vertex_ai_location,
            vertex_project=vertex_ai_project,
            vertex_credentials=vertex_credentials,
            logging_obj=logging,
            acompletion=acompletion,
        )

        if "stream" in optional_params and optional_params["stream"] is True and acompletion is False:
            return CustomStreamWrapper(
                model_response,
                model,
                custom_llm_provider="vertex_ai",
                logging_obj=logging,
            )
    return model_response  # pyright: ignore[reportReturnType]  # provider SDK return type is broader than the dispatch contract


def complete_gdc(ctx: CompletionDispatchContext) -> CompletionDispatchResult:
    acompletion: Final = ctx.acompletion
    api_base = ctx.api_base
    api_key = ctx.api_key
    client: Final = dispatch_client_http(ctx)
    custom_llm_provider: Final = ctx.custom_llm_provider
    headers: Final = ctx.headers
    litellm_params: Final = ctx.litellm_params
    logging: Final = ctx.logging
    messages: Final = ctx.messages
    model: Final = ctx.model
    model_response: Final = ctx.model_response
    optional_params: Final = ctx.optional_params
    stream: Final = ctx.stream
    timeout: Final = ctx.timeout

    api_key = api_key or litellm.gdc_key or get_secret_str("GDC_API_KEY") or litellm.api_key
    api_base = api_base or litellm.gdc_api_base or get_secret_str("GDC_API_BASE") or litellm.api_base

    return base_llm_http_handler.completion(
        model=model,
        messages=messages,
        headers=headers,
        model_response=model_response,
        api_key=api_key,
        api_base=api_base,
        acompletion=acompletion,
        logging_obj=logging,
        optional_params=optional_params,
        litellm_params=litellm_params,
        timeout=timeout,
        client=client,
        custom_llm_provider=custom_llm_provider,
        encoding=get_encoding(),
        stream=stream,
        provider_config=gdc_transformation,
    )
