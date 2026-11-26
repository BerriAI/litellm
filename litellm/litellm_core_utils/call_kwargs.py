"""Typed reads of the values callers and the `@client` decorator pass through `**kwargs`."""

from aiohttp import ClientSession

from litellm.litellm_core_utils.core_helpers import has_attributes, require_attributes
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


def logging_obj_of(value: object) -> Logging:
    return require_attributes(value, Logging, "update_from_kwargs")


def http_client_of(value: object) -> HTTPHandler | AsyncHTTPHandler | None:
    return value if isinstance(value, (HTTPHandler, AsyncHTTPHandler)) else None


def shared_session_of(value: object) -> ClientSession | None:
    return value if has_attributes(value, ClientSession, "closed") else None


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
