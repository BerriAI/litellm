import asyncio

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


class RequestDisconnectMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        disconnect_task = asyncio.create_task(self.check_disconnect(request))

        try:
            response = await call_next(request)
            return response
        except asyncio.CancelledError:
            # The request was disconnected
            return Response(status_code=499, content="Client Disconnected")
        finally:
            disconnect_task.cancel()

    async def check_disconnect(self, request: Request):
        while True:
            if await request.is_disconnected():
                # Cancel the ongoing operation
                _current_task = asyncio.current_task()
                if _current_task:
                    _current_task.cancel()
                return
            await asyncio.sleep(1)  # Check every 1 second
