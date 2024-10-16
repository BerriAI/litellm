import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, List, Literal, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.azure_ai.rerank import AzureAIRerank
from litellm.llms.cohere.rerank import CohereRerank
from litellm.llms.together_ai.rerank import TogetherAIRerank
from litellm.secret_managers.main import get_secret
from litellm.types.rerank import RerankRequest, RerankResponse
from litellm.types.router import *
from litellm.utils import client, exception_type, supports_httpx_timeout

####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
cohere_rerank = CohereRerank()
together_rerank = TogetherAIRerank()
azure_ai_rerank = AzureAIRerank()
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
def rerank(
    model: str,
    query: str,
    documents: List[Union[str, Dict[str, Any]]],
    custom_llm_provider: Optional[Literal["cohere", "together_ai", "azure_ai"]] = None,
    top_n: Optional[int] = None,
    rank_fields: Optional[List[str]] = None,
    return_documents: Optional[bool] = True,
    max_chunks_per_doc: Optional[int] = None,
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
    try:
        _is_async = kwargs.pop("arerank", False) is True
        optional_params = GenericLiteLLMParams(**kwargs)

        model, _custom_llm_provider, dynamic_api_key, dynamic_api_base = (
            litellm.get_llm_provider(
                model=model,
                custom_llm_provider=custom_llm_provider,
                api_base=optional_params.api_base,
                api_key=optional_params.api_key,
            )
        )

        model_params_dict = {
            "top_n": top_n,
            "rank_fields": rank_fields,
            "return_documents": return_documents,
            "max_chunks_per_doc": max_chunks_per_doc,
            "documents": documents,
        }

        litellm_logging_obj.update_environment_variables(
            model=model,
            user=user,
            optional_params=model_params_dict,
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
        if _custom_llm_provider == "cohere":
            # Implement Cohere rerank logic
            api_key: Optional[str] = (
                dynamic_api_key
                or optional_params.api_key
                or litellm.cohere_key
                or get_secret("COHERE_API_KEY")  # type: ignore
                or get_secret("CO_API_KEY")  # type: ignore
                or litellm.api_key
            )

            if api_key is None:
                raise ValueError(
                    "Cohere API key is required, please set 'COHERE_API_KEY' in your environment"
                )

            api_base: Optional[str] = (
                dynamic_api_base
                or optional_params.api_base
                or litellm.api_base
                or get_secret("COHERE_API_BASE")  # type: ignore
                or "https://api.cohere.com/v1/rerank"
            )

            if api_base is None:
                raise Exception(
                    "Invalid api base. api_base=None. Set in call or via `COHERE_API_BASE` env var."
                )

            headers = headers or litellm.headers or {}

            response = cohere_rerank.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=top_n,
                rank_fields=rank_fields,
                return_documents=return_documents,
                max_chunks_per_doc=max_chunks_per_doc,
                api_key=api_key,
                api_base=api_base,
                _is_async=_is_async,
                headers=headers,
                litellm_logging_obj=litellm_logging_obj,
            )
        elif _custom_llm_provider == "azure_ai":
            api_base = (
                dynamic_api_base  # for deepinfra/perplexity/anyscale/groq/friendliai we check in get_llm_provider and pass in the api base from there
                or optional_params.api_base
                or litellm.api_base
                or get_secret("AZURE_AI_API_BASE")  # type: ignore
            )
            # set API KEY
            api_key = (
                dynamic_api_key
                or litellm.api_key  # for deepinfra/perplexity/anyscale/friendliai we check in get_llm_provider and pass in the api key from there
                or litellm.openai_key
                or get_secret("AZURE_AI_API_KEY")  # type: ignore
            )

            headers = headers or litellm.headers or {}

            if api_key is None:
                raise ValueError(
                    "Azure AI API key is required, please set 'AZURE_AI_API_KEY' in your environment"
                )

            if api_base is None:
                raise Exception(
                    "Azure AI API Base is required. api_base=None. Set in call or via `AZURE_AI_API_BASE` env var."
                )

            ## LOAD CONFIG - if set
            config = litellm.OpenAIConfig.get_config()
            for k, v in config.items():
                if (
                    k not in optional_params
                ):  # completion(top_k=3) > openai_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            response = azure_ai_rerank.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=top_n,
                rank_fields=rank_fields,
                return_documents=return_documents,
                max_chunks_per_doc=max_chunks_per_doc,
                api_key=api_key,
                api_base=api_base,
                _is_async=_is_async,
                headers=headers,
                litellm_logging_obj=litellm_logging_obj,
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

        else:
            raise ValueError(f"Unsupported provider: {_custom_llm_provider}")

        # Placeholder return
        return response
    except Exception as e:
        verbose_logger.error(f"Error in rerank: {str(e)}")
        raise exception_type(
            model=model, custom_llm_provider=custom_llm_provider, original_exception=e
        )
