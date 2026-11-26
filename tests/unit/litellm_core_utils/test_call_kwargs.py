from typing import Final
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientSession

from litellm.litellm_core_utils.call_kwargs import (
    http_client_of,
    logging_obj_of,
    optional_str,
    shared_session_of,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


def test_logging_obj_of_returns_the_same_logging_object() -> None:
    logging_obj: Final = MagicMock(spec=Logging)

    assert logging_obj_of(logging_obj) is logging_obj


def test_logging_obj_of_missing_object_raises_the_attribute_error_its_first_use_raised() -> None:
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'update_from_kwargs'"):
        logging_obj_of(None)


@pytest.mark.parametrize("handler_class", [HTTPHandler, AsyncHTTPHandler])
def test_http_client_of_keeps_litellm_http_handlers(handler_class: type[HTTPHandler] | type[AsyncHTTPHandler]) -> None:
    handler: Final = handler_class()

    assert http_client_of(handler) is handler


@pytest.mark.parametrize("value", [None, "not-a-client", object()])
def test_http_client_of_drops_values_the_handlers_would_replace(value: object) -> None:
    assert http_client_of(value) is None


@pytest.mark.asyncio
async def test_shared_session_of_keeps_an_aiohttp_session() -> None:
    async with ClientSession() as session:
        assert shared_session_of(session) is session


def test_shared_session_of_keeps_a_duck_typed_session() -> None:
    session: Final = MagicMock(closed=False)

    assert shared_session_of(session) is session


@pytest.mark.parametrize("value", [None, "not-a-session", 7])
def test_shared_session_of_drops_values_without_a_closed_flag(value: object) -> None:
    assert shared_session_of(value) is None


def test_optional_str_keeps_strings_including_empty() -> None:
    assert optional_str("user-1") == "user-1"
    assert optional_str("") == ""


@pytest.mark.parametrize("value", [None, 7, ["user-1"], {"id": "user-1"}])
def test_optional_str_drops_non_strings(value: object) -> None:
    assert optional_str(value) is None
