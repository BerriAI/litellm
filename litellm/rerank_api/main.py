import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, List, Literal, Optional, Union

import litellm
from litellm import get_secret
from litellm._logging import verbose_logger
from litellm.llms.cohere.rerank import CohereRerank
from litellm.types.router import *
from litellm.utils import supports_httpx_timeout

from .types import RerankRequest, RerankResponse

####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
cohere_rerank = CohereRerank()
#################################################


async def arerank(
    model: str,
    query: str,
    documents: List[str],
    custom_llm_provider: Literal["cohere", "together_ai"] = "cohere",
    top_n: int = 3,
    **kwargs,
) -> Dict[str, Any]:
    """
    Async: Reranks a list of documents based on their relevance to the query
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["arerank"] = True

        func = partial(
            rerank, model, query, documents, custom_llm_provider, top_n, **kwargs
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


def rerank(
    model: str,
    query: str,
    documents: List[str],
    custom_llm_provider: Literal["cohere", "together_ai"] = "cohere",
    top_n: int = 3,
    **kwargs,
) -> Union[RerankResponse, Coroutine[Any, Any, RerankResponse]]:
    """
    Reranks a list of documents based on their relevance to the query
    """
    try:
        _is_async = kwargs.pop("arerank", False) is True
        optional_params = GenericLiteLLMParams(**kwargs)

        # Implement rerank logic here based on the custom_llm_provider
        if custom_llm_provider == "cohere":
            # Implement Cohere rerank logic
            cohere_key = (
                optional_params.api_key
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
                or "https://api.cohere.ai/v1/generate"
            )

            headers: Dict = litellm.headers or {}

            response = cohere_rerank.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=top_n,
                api_key=cohere_key,
            )
            pass
        elif custom_llm_provider == "together_ai":
            # Implement Together AI rerank logic
            pass
        else:
            raise ValueError(f"Unsupported provider: {custom_llm_provider}")

        # Placeholder return
        return response
    except Exception as e:
        verbose_logger.error(f"Error in rerank: {str(e)}")
        raise e
