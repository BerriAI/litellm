import asyncio
import copy
import json
import os
from copy import deepcopy
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import httpx

import litellm
from litellm.llms.cohere.embed import embedding as cohere_embedding
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.OpenAI.openai import OpenAIChatCompletion

from .cohere_transformation import AzureAICohereConfig


class AzureAIEmbedding(OpenAIChatCompletion):
    def embedding(
        self,
        model: str,
        input: List,
        timeout: float,
        logging_obj,
        model_response: litellm.EmbeddingResponse,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        client=None,
        aembedding=None,
    ):
        response = super().embedding(
            model,
            input,
            timeout,
            logging_obj,
            model_response,
            optional_params,
            api_key,
            api_base,
            client,
            aembedding,
        )

        if not asyncio.iscoroutine(response):
            response = AzureAICohereConfig()._transform_response(response=response)  # type: ignore

        return response
