"""
This is a modification of code from: https://github.com/SecretiveShell/MCP-Bridge/blob/master/mcp_bridge/mcp_server/sse_transport.py

Credit to the maintainers of SecretiveShell for their SSE Transport implementation

"""

from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import quote
from uuid import UUID, uuid4

import anyio
import mcp.types as types
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from fastapi.requests import Request
from fastapi.responses import Response
from pydantic import ValidationError
from sse_starlette import EventSourceResponse
from starlette.types import Receive, Scope, Send

from litellm._logging import verbose_logger


class SseServerTransport:
    """
    SSE server transport for MCP. This class provides _two_ ASGI applications,
    suitable to be used with a framework like Starlette and a server like Hypercorn:

        1. connect_sse() is an ASGI application which receives incoming GET requests,
           and sets up a new SSE stream to send server messages to the client.
        2. handle_post_message() is an ASGI application which receives incoming POST
           requests, which should contain client messages that link to a
           previously-established SSE session.
    """

    _endpoint: str
    _read_stream_writers: dict[
        UUID, MemoryObjectSendStream[types.JSONRPCMessage | Exception]
    ]

    def __init__(self, endpoint: str) -> None:
        """
        Creates a new SSE server transport, which will direct the client to POST
        messages to the relative or absolute URL given.
        """

        super().__init__()
        self._endpoint = endpoint
        self._read_stream_writers = {}
        verbose_logger.debug(
            f"SseServerTransport initialized with endpoint: {endpoint}"
        )

    @asynccontextmanager
    async def connect_sse(self, request: Request):
        if request.scope["type"] != "http":
            verbose_logger.error("connect_sse received non-HTTP request")
            raise ValueError("connect_sse can only handle HTTP requests")

        verbose_logger.debug("Setting up SSE connection")
        read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
        read_stream_writer: MemoryObjectSendStream[types.JSONRPCMessage | Exception]

        write_stream: MemoryObjectSendStream[types.JSONRPCMessage]
        write_stream_reader: MemoryObjectReceiveStream[types.JSONRPCMessage]

        read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
        write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

        session_id = uuid4()
        session_uri = f"{quote(self._endpoint)}?session_id={session_id.hex}"
        self._read_stream_writers[session_id] = read_stream_writer
        verbose_logger.debug(f"Created new session with ID: {session_id}")

        sse_stream_writer: MemoryObjectSendStream[dict[str, Any]]
        sse_stream_reader: MemoryObjectReceiveStream[dict[str, Any]]
        sse_stream_writer, sse_stream_reader = anyio.create_memory_object_stream(
            0, dict[str, Any]
        )

        async def sse_writer():
            verbose_logger.debug("Starting SSE writer")
            async with sse_stream_writer, write_stream_reader:
                await sse_stream_writer.send({"event": "endpoint", "data": session_uri})
                verbose_logger.debug(f"Sent endpoint event: {session_uri}")

                async for message in write_stream_reader:
                    verbose_logger.debug(f"Sending message via SSE: {message}")
                    await sse_stream_writer.send(
                        {
                            "event": "message",
                            "data": message.model_dump_json(
                                by_alias=True, exclude_none=True
                            ),
                        }
                    )

        async with anyio.create_task_group() as tg:
            response = EventSourceResponse(
                content=sse_stream_reader, data_sender_callable=sse_writer
            )
            verbose_logger.debug("Starting SSE response task")
            tg.start_soon(response, request.scope, request.receive, request._send)

            verbose_logger.debug("Yielding read and write streams")
            yield (read_stream, write_stream)

    async def handle_post_message(
        self, scope: Scope, receive: Receive, send: Send
    ) -> Response:
        verbose_logger.debug("Handling POST message")
        request = Request(scope, receive)

        session_id_param = request.query_params.get("session_id")
        if session_id_param is None:
            verbose_logger.warning("Received request without session_id")
            response = Response("session_id is required", status_code=400)
            return response

        try:
            session_id = UUID(hex=session_id_param)
            verbose_logger.debug(f"Parsed session ID: {session_id}")
        except ValueError:
            verbose_logger.warning(f"Received invalid session ID: {session_id_param}")
            response = Response("Invalid session ID", status_code=400)
            return response

        writer = self._read_stream_writers.get(session_id)
        if not writer:
            verbose_logger.warning(f"Could not find session for ID: {session_id}")
            response = Response("Could not find session", status_code=404)
            return response

        json = await request.json()
        verbose_logger.debug(f"Received JSON: {json}")

        try:
            message = types.JSONRPCMessage.model_validate(json)
            verbose_logger.debug(f"Validated client message: {message}")
        except ValidationError as err:
            verbose_logger.error(f"Failed to parse message: {err}")
            response = Response("Could not parse message", status_code=400)
            await writer.send(err)
            return response

        verbose_logger.debug(f"Sending message to writer: {message}")
        response = Response("Accepted", status_code=202)
        await writer.send(message)
        return response
