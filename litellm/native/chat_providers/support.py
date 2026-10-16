from collections.abc import Iterable, Mapping
from functools import cache

import openai

import litellm
from litellm._lazy_imports import get_default_encoding
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.completion import CompletionDispatchContext


def dispatch_metadata(ctx: CompletionDispatchContext) -> Mapping[str, object] | None:
    return ctx.metadata


def dispatch_client_http(ctx: CompletionDispatchContext) -> HTTPHandler | AsyncHTTPHandler | None:
    return ctx.client


def dispatch_client_azure(
    ctx: CompletionDispatchContext,
) -> openai.AzureOpenAI | openai.AsyncAzureOpenAI | HTTPHandler | AsyncHTTPHandler | None:
    return ctx.client


def dispatch_client_openai(ctx: CompletionDispatchContext) -> openai.OpenAI | openai.AsyncOpenAI | None:
    return ctx.client


def provider_config_items(config: Mapping[str, object]) -> Iterable[tuple[str, object]]:
    return config.items()


def print_verbose(print_statement: object) -> None:
    try:
        verbose_logger.debug(print_statement)
        if litellm.set_verbose:
            print(print_statement)  # noqa: T201
    except Exception:
        pass


@cache
def get_encoding() -> Tokenizer:
    return get_default_encoding()
