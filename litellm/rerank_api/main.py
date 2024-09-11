import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, List, Literal, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.llms.cohere.rerank import CohereRerank
from litellm.llms.togetherai.rerank import TogetherAIRerank
from litellm.secret_managers.main import get_secret
from litellm.types.router import *
from litellm.utils import client, supports_httpx_timeout

from .types import RerankRequest, RerankResponse

####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
cohere_rerank = CohereRerank()
together_rerank = TogetherAIRerank()
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
    custom_llm_provider: Optional[Literal["cohere", "together_ai"]] = None,
    top_n: Optional[int] = None,
    rank_fields: Optional[List[str]] = None,
    return_documents: Optional[bool] = True,
    max_chunks_per_doc: Optional[int] = None,
    **kwargs,
) -> Union[RerankResponse, Coroutine[Any, Any, RerankResponse]]:
    """
    Reranks a list of documents based on their relevance to the query
    """
    try:
        _is_async = kwargs.pop("arerank", False) is True
        optional_params = GenericLiteLLMParams(**kwargs)

        model, _custom_llm_provider, dynamic_api_key, api_base = (
            litellm.get_llm_provider(
                model=model,
                custom_llm_provider=custom_llm_provider,
                api_base=optional_params.api_base,
                api_key=optional_params.api_key,
            )
        )

        # Implement rerank logic here based on the custom_llm_provider
        if _custom_llm_provider == "cohere":
            # Implement Cohere rerank logic
            cohere_key = (
                dynamic_api_key
                or optional_params.api_key
                or litellm.cohere_key
                or get_secret("COHERE_API_KEY")
                or get_secret("CO_API_KEY")
                or litellm.api_key
            )

            if cohere_key is None:
                raise ValueError(
                    "Cohere API key is required, please set 'COHERE_API_KEY' in your environment"
                )

            api_base = (
                optional_params.api_base
                or litellm.api_base
                or get_secret("COHERE_API_BASE")
                or "https://api.cohere.com/v1/rerank"
            )

            headers: Dict = litellm.headers or {}

            response = cohere_rerank.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=top_n,
                rank_fields=rank_fields,
                return_documents=return_documents,
                max_chunks_per_doc=max_chunks_per_doc,
                api_key=cohere_key,
                api_base=api_base,
                _is_async=_is_async,
            )
            pass
        elif _custom_llm_provider == "together_ai":
            # Implement Together AI rerank logic
            together_key = (
                dynamic_api_key
                or optional_params.api_key
                or litellm.togetherai_api_key
                or get_secret("TOGETHERAI_API_KEY")
                or litellm.api_key
            )

            if together_key is None:
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
                api_key=together_key,
                _is_async=_is_async,
            )

        else:
            raise ValueError(f"Unsupported provider: {_custom_llm_provider}")

        # Placeholder return
        return response
    except Exception as e:
        verbose_logger.error(f"Error in rerank: {str(e)}")
        raise e
