"""
This module is used to pass through requests to the LLM APIs.
"""

import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, List, Literal, Optional, Union
from urllib.parse import urlencode

import httpx
from httpx._types import CookieTypes, HeaderTypes, QueryParamTypes, RequestFiles

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import client

from .utils import BasePassthroughUtils

# @client
# async def allm_passthrough_route(
#     model: str,
#     query: str,
#     documents: List[Union[str, Dict[str, Any]]],
#     custom_llm_provider: Optional[Literal["cohere", "together_ai"]] = None,
#     top_n: Optional[int] = None,
#     rank_fields: Optional[List[str]] = None,
#     return_documents: Optional[bool] = None,
#     max_chunks_per_doc: Optional[int] = None,
#     **kwargs,
# ) -> Union[httpx.Response, Coroutine[Any, Any, httpx.Response]]:
#     """
#     Async: Reranks a list of documents based on their relevance to the query
#     """
#     try:
#         loop = asyncio.get_event_loop()
#         kwargs["arerank"] = True

#         func = partial(
#             rerank,
#             model,
#             query,
#             documents,
#             custom_llm_provider,
#             top_n,
#             rank_fields,
#             return_documents,
#             max_chunks_per_doc,
#             **kwargs,
#         )

#         ctx = contextvars.copy_context()
#         func_with_context = partial(ctx.run, func)
#         init_response = await loop.run_in_executor(None, func_with_context)

#         if asyncio.iscoroutine(init_response):
#             response = await init_response
#         else:
#             response = init_response
#         return response
#     except Exception as e:
#         raise e


@client
def llm_passthrough_route(
    *,
    method: str,
    request_url: str,
    target_api_base: str,
    request_query_params: Optional[dict] = None,
    request_headers: Optional[dict] = None,
    allm_passthrough_route: bool = False,
    stream: bool = False,
    content: Optional[Any] = None,
    data: Optional[dict] = None,
    files: Optional[RequestFiles] = None,
    json: Optional[Any] = None,
    params: Optional[QueryParamTypes] = None,
    headers: Optional[dict] = None,
    cookies: Optional[CookieTypes] = None,
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    **kwargs,
) -> Union[httpx.Response, Coroutine[Any, Any, httpx.Response]]:
    """
    Pass through requests to the LLM APIs.

    Step 1. Build the request
    Step 2. Send the request
    Step 3. Return the response
    """
    if client is None:
        if allm_passthrough_route:
            client = litellm.module_level_aclient
        else:
            client = litellm.module_level_client

    url = httpx.URL(request_url)

    if request_query_params:
        # Create a new URL with the merged query params
        url = url.copy_with(query=urlencode(request_query_params).encode("ascii"))

    headers = BasePassthroughUtils.forward_headers_from_request(
        request_headers=request_headers or {},
        headers=headers if headers else {},
        forward_headers=True,
    )

    request = client.client.build_request(
        method=method,
        url=target_api_base,
        content=content,
        data=data,
        files=files,
        json=json,
        params=params,
        headers=headers,
        cookies=cookies,
    )

    response = client.client.send(request=request, stream=stream)
    return response
