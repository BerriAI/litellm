import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, List, Literal, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.llms.bedrock.rerank.handler import BedrockRerankHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.together_ai.rerank.handler import TogetherAIRerank
from litellm.rerank_api.rerank_utils import get_optional_rerank_params
from litellm.secret_managers.main import get_secret, get_secret_str
from litellm.types.rerank import OptionalRerankParams, RerankResponse
from litellm.types.router import *
from litellm.utils import ProviderConfigManager, client, exception_type

####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
together_rerank = TogetherAIRerank()
bedrock_rerank = BedrockRerankHandler()
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


@client
async def arerank(
    model: str,
    query: str,
    documents: List[Union[str, Dict[str, Any]]],
    custom_llm_provider: Optional[Literal["cohere", "together_ai"]] = None,
    top_n: Optional[int] = None,
    rank_fields: Optional[List[str]] = None,
    return_documents: Optional[bool] = None,
    max_chunks_per_doc: Optional[int] = None,
    **kwargs,
) -> Union[RerankResponse, Coroutine[Any, Any, RerankResponse]]:
    """
    Async: Reranks a list of documents based on their relevance to the query
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["arerank"] = True

        func = partial(
            rerank,
            model,
            query,
            documents,
            custom_llm_provider,
            top_n,
            rank_fields,
            return_documents,
            max_chunks_per_doc,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:
        raise e


@client
def rerank(  # noqa: PLR0915
    model: str,
    query: str,
    documents: List[Union[str, Dict[str, Any]]],
    custom_llm_provider: Optional[
        Literal["cohere", "together_ai", "azure_ai", "infinity", "litellm_proxy"]
    ] = None,
    top_n: Optional[int] = None,
    rank_fields: Optional[List[str]] = None,
    return_documents: Optional[bool] = True,
    max_chunks_per_doc: Optional[int] = None,
    max_tokens_per_doc: Optional[int] = None,
    **kwargs,
) -> Union[RerankResponse, Coroutine[Any, Any, RerankResponse]]:
    """
    Reranks a list of documents based on their relevance to the query
    """
    headers: Optional[dict] = kwargs.get("headers")  # type: ignore
    litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
    litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
    proxy_server_request = kwargs.get("proxy_server_request", None)
    model_info = kwargs.get("model_info", None)
    metadata = kwargs.get("metadata", {})
    user = kwargs.get("user", None)
    client = kwargs.get("client", None)
    try:
        _is_async = kwargs.pop("arerank", False) is True
        optional_params = GenericLiteLLMParams(**kwargs)
        # Params that are unique to specific versions of the client for the rerank call
        unique_version_params = {
            "max_chunks_per_doc": max_chunks_per_doc,
            "max_tokens_per_doc": max_tokens_per_doc,
        }
        present_version_params = [
            k for k, v in unique_version_params.items() if v is not None
        ]

        model, _custom_llm_provider, dynamic_api_key, dynamic_api_base = (
            litellm.get_llm_provider(
                model=model,
                custom_llm_provider=custom_llm_provider,
                api_base=optional_params.api_base,
                api_key=optional_params.api_key,
            )
        )

        rerank_provider_config: BaseRerankConfig = (
            ProviderConfigManager.get_provider_rerank_config(
                model=model,
                provider=litellm.LlmProviders(_custom_llm_provider),
                api_base=optional_params.api_base,
                present_version_params=present_version_params,
            )
        )

        optional_rerank_params: OptionalRerankParams = get_optional_rerank_params(
            rerank_provider_config=rerank_provider_config,
            model=model,
            drop_params=kwargs.get("drop_params") or litellm.drop_params or False,
            query=query,
            documents=documents,
            custom_llm_provider=_custom_llm_provider,
            top_n=top_n,
            rank_fields=rank_fields,
            return_documents=return_documents,
            max_chunks_per_doc=max_chunks_per_doc,
            max_tokens_per_doc=max_tokens_per_doc,
            non_default_params=kwargs,
        )

        if isinstance(optional_params.timeout, str):
            optional_params.timeout = float(optional_params.timeout)

        model_response = RerankResponse()

        litellm_logging_obj.update_environment_variables(
            model=model,
            user=user,
            optional_params=dict(optional_rerank_params),
            litellm_params={
                "litellm_call_id": litellm_call_id,
                "proxy_server_request": proxy_server_request,
                "model_info": model_info,
                "metadata": metadata,
                "preset_cache_key": None,
                "stream_response": {},
                **optional_params.model_dump(exclude_unset=True),
            },
            custom_llm_provider=_custom_llm_provider,
        )

        # Implement rerank logic here based on the custom_llm_provider
        if _custom_llm_provider == "cohere" or _custom_llm_provider == "litellm_proxy":
            # Implement Cohere rerank logic
            api_key: Optional[str] = (
                dynamic_api_key or optional_params.api_key or litellm.api_key
            )

            api_base: Optional[str] = (
                dynamic_api_base
                or optional_params.api_base
                or litellm.api_base
                or get_secret("COHERE_API_BASE")  # type: ignore
                or "https://api.cohere.com"
            )

            if api_base is None:
                raise Exception(
                    "Invalid api base. api_base=None. Set in call or via `COHERE_API_BASE` env var."
                )
            response = base_llm_http_handler.rerank(
                model=model,
                custom_llm_provider=_custom_llm_provider,
                provider_config=rerank_provider_config,
                optional_rerank_params=optional_rerank_params,
                logging_obj=litellm_logging_obj,
                timeout=optional_params.timeout,
                api_key=api_key,
                api_base=api_base,
                _is_async=_is_async,
                headers=headers or litellm.headers or {},
                client=client,
                model_response=model_response,
            )
        elif _custom_llm_provider == "azure_ai":
            api_base = (
                dynamic_api_base  # for deepinfra/perplexity/anyscale/groq/friendliai we check in get_llm_provider and pass in the api base from there
                or optional_params.api_base
                or litellm.api_base
                or get_secret("AZURE_AI_API_BASE")  # type: ignore
            )
            response = base_llm_http_handler.rerank(
                model=model,
                custom_llm_provider=_custom_llm_provider,
                optional_rerank_params=optional_rerank_params,
                provider_config=rerank_provider_config,
                logging_obj=litellm_logging_obj,
                timeout=optional_params.timeout,
                api_key=dynamic_api_key or optional_params.api_key,
                api_base=api_base,
                _is_async=_is_async,
                headers=headers or litellm.headers or {},
                client=client,
                model_response=model_response,
            )
        elif _custom_llm_provider == "infinity":
            # Implement Infinity rerank logic
            api_key = dynamic_api_key or optional_params.api_key or litellm.api_key

            api_base = (
                dynamic_api_base
                or optional_params.api_base
                or litellm.api_base
                or get_secret_str("INFINITY_API_BASE")
            )

            if api_base is None:
                raise Exception(
                    "Invalid api base. api_base=None. Set in call or via `INFINITY_API_BASE` env var."
                )

            response = base_llm_http_handler.rerank(
                model=model,
                custom_llm_provider=_custom_llm_provider,
                provider_config=rerank_provider_config,
                optional_rerank_params=optional_rerank_params,
                logging_obj=litellm_logging_obj,
                timeout=optional_params.timeout,
                api_key=dynamic_api_key or optional_params.api_key,
                api_base=api_base,
                _is_async=_is_async,
                headers=headers or litellm.headers or {},
                client=client,
                model_response=model_response,
            )
        elif _custom_llm_provider == "together_ai":
            # Implement Together AI rerank logic
            api_key = (
                dynamic_api_key
                or optional_params.api_key
                or litellm.togetherai_api_key
                or get_secret("TOGETHERAI_API_KEY")  # type: ignore
                or litellm.api_key
            )

            if api_key is None:
                raise ValueError(
                    "TogetherAI API key is required, please set 'TOGETHERAI_API_KEY' in your environment"
                )

            response = together_rerank.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=top_n,
                rank_fields=rank_fields,
                return_documents=return_documents,
                max_chunks_per_doc=max_chunks_per_doc,
                api_key=api_key,
                _is_async=_is_async,
            )
        elif _custom_llm_provider == "jina_ai":

            if dynamic_api_key is None:
                raise ValueError(
                    "Jina AI API key is required, please set 'JINA_AI_API_KEY' in your environment"
                )

            api_base = (
                dynamic_api_base
                or optional_params.api_base
                or litellm.api_base
                or get_secret("BEDROCK_API_BASE")  # type: ignore
            )

            response = base_llm_http_handler.rerank(
                model=model,
                custom_llm_provider=_custom_llm_provider,
                optional_rerank_params=optional_rerank_params,
                logging_obj=litellm_logging_obj,
                provider_config=rerank_provider_config,
                timeout=optional_params.timeout,
                api_key=dynamic_api_key or optional_params.api_key,
                api_base=api_base,
                _is_async=_is_async,
                headers=headers or litellm.headers or {},
                client=client,
                model_response=model_response,
            )
        elif _custom_llm_provider == "bedrock":
            api_base = (
                dynamic_api_base
                or optional_params.api_base
                or litellm.api_base
                or get_secret("BEDROCK_API_BASE")  # type: ignore
            )

            response = bedrock_rerank.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=top_n,
                rank_fields=rank_fields,
                return_documents=return_documents,
                max_chunks_per_doc=max_chunks_per_doc,
                _is_async=_is_async,
                optional_params=optional_params.model_dump(exclude_unset=True),
                api_base=api_base,
                logging_obj=litellm_logging_obj,
                client=client,
            )
        else:
            raise ValueError(f"Unsupported provider: {_custom_llm_provider}")

        # Placeholder return
        return response
    except Exception as e:
        verbose_logger.error(f"Error in rerank: {str(e)}")
        raise exception_type(
            model=model, custom_llm_provider=custom_llm_provider, original_exception=e
        )
