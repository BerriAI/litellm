import asyncio
import json
from collections.abc import Callable
from typing import Final, Literal

import httpx
import pytest
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from typing_extensions import ReadOnly, TypedDict

import litellm
from integration.providers._vertex_batch_rows import (
    BOM,
    BUCKET,
    LOCATION,
    PROJECT,
    chat_row,
    encoded,
    is_gcs_upload,
    jsonl,
    roleless_row,
    stored_object,
    unconvertible_line_message,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.vertex_ai.common_utils import VertexAIError

Client = Literal["sync", "async_aiohttp", "async_httpx"]
CLIENTS: Final[tuple[Client, ...]] = ("sync", "async_aiohttp", "async_httpx")
_TOKEN_REPLY: Final = Reply(
    body=json.dumps({"access_token": "scripted-token", "token_type": "Bearer", "expires_in": 3600}).encode()
)


def _peer(request: Request) -> Reply:
    if request.target == "/_oauth/token":
        return _TOKEN_REPLY
    assert is_gcs_upload(request), (request.method, request.target)
    return stored_object(request)


class _VertexKwargs(TypedDict):
    custom_llm_provider: ReadOnly[Literal["vertex_ai"]]
    api_base: ReadOnly[str]
    vertex_project: ReadOnly[str]
    vertex_location: ReadOnly[str]
    vertex_credentials: ReadOnly[str]
    gcs_bucket_name: ReadOnly[str]


def _vertex_kwargs(wire: Wire) -> _VertexKwargs:
    return {
        "custom_llm_provider": "vertex_ai",
        "api_base": wire.url,
        "vertex_project": PROJECT,
        "vertex_location": LOCATION,
        "vertex_credentials": service_account_json(PROJECT, wire.url),
        "gcs_bucket_name": BUCKET,
    }


async def _create_file(client: Client, wire: Wire, content: bytes) -> object:
    file: Final = ("rows.jsonl", content, "application/jsonl")
    match client:
        case "sync":
            return await asyncio.to_thread(litellm.create_file, file=file, purpose="batch", **_vertex_kwargs(wire))
        case "async_aiohttp":
            return await litellm.acreate_file(file=file, purpose="batch", **_vertex_kwargs(wire))
        case "async_httpx":
            handler: Final = AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport(trust_env=False))
            return await litellm.acreate_file(file=file, purpose="batch", client=handler, **_vertex_kwargs(wire))


def _good(custom_id: str) -> bytes:
    return encoded(chat_row(custom_id, "gemini-2.5-flash", "fine"))


ROWS: Final[tuple[tuple[str, Callable[[], bytes], str], ...]] = (
    (
        "malformed_json_line_3",
        lambda: jsonl(_good("first"), b"", b'{"custom_id": "bad", '),
        "Batch input file line 3 is not valid JSON: ",
    ),
    (
        "invalid_utf8_line_3",
        lambda: jsonl(_good("first"), b"", b'{"custom_id": "\xff\xfe"}'),
        "Batch input file line 3 is not valid UTF-8: ",
    ),
    (
        "malformed_json_line_1",
        lambda: jsonl(b'{"custom_id": "bad", ', _good("second")),
        "Batch input file line 1 is not valid JSON: ",
    ),
    (
        "roleless_row_line_4",
        lambda: jsonl(BOM + _good("first"), b"", b"  ", encoded(roleless_row("bad", "gemini-2.5-flash"))),
        unconvertible_line_message(4) + "missing field 'role'",
    ),
)


@pytest.mark.parametrize(
    ("content", "message"), tuple(row[1:] for row in ROWS), ids=tuple(row[0] for row in ROWS)
)
@pytest.mark.parametrize("client", CLIENTS)
async def test_sdk_create_file_names_the_file_line_vertex_cannot_read(
    client: Client, content: Callable[[], bytes], message: str
) -> None:
    with wire_server(_peer) as wire:
        with pytest.raises(VertexAIError) as raised:
            await _create_file(client, wire, content())
        assert raised.value.status_code == 400
        assert message in str(raised.value), str(raised.value)
        assert not any(is_gcs_upload(request) for request in wire.drain())


async def test_router_create_file_names_the_line_after_rewriting_each_rows_model() -> None:
    with wire_server(_peer) as wire:
        router: Final = litellm.Router(
            model_list=[
                {
                    "model_name": "vertex-batch",
                    "litellm_params": {"model": "vertex_ai/gemini-2.5-flash", **_vertex_kwargs(wire)},
                }
            ]
        )
        content: Final = jsonl(
            BOM + encoded(chat_row("first", "vertex-batch", "one")),
            b"",
            b"  ",
            encoded(roleless_row("bad", "vertex-batch")),
        )
        with pytest.raises(VertexAIError) as raised:
            await router.acreate_file(
                model="vertex-batch", file=("rows.jsonl", content, "application/jsonl"), purpose="batch"
            )
        assert raised.value.status_code == 400
        assert unconvertible_line_message(4) + "missing field 'role'" in str(raised.value), str(raised.value)
        assert not any(is_gcs_upload(request) for request in wire.drain())
