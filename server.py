import base64

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

mcp = FastMCP("Edit preview verification", host="0.0.0.0", port=8080, stateless_http=True, json_response=True)

@mcp.tool()
def echo(message: str) -> str:
    """Return the supplied message to verify the connection."""
    return message

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        expected = "Basic " + base64.b64encode(b"preview:correct").decode()
        if request.headers.get("authorization") != expected and request.headers.get("x-preview-key") != "correct":
            return JSONResponse({"error": "Basic authentication required"}, status_code=403)
        return await call_next(request)

app = mcp.streamable_http_app()
app.add_middleware(AuthMiddleware)
uvicorn.run(app, host="0.0.0.0", port=8080, access_log=False)
