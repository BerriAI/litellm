from dataclasses import dataclass

from starlette.requests import Request
from starlette.responses import Response

from mcp.server.auth.json_response import PydanticJSONResponse
from mcp.shared.auth import OAuthMetadata, ProtectedResourceMetadata


@dataclass
class MetadataHandler:
    metadata: OAuthMetadata

    async def handle(self, request: Request) -> Response:
        return PydanticJSONResponse(
            content=self.metadata,
            headers={"Cache-Control": "public, max-age=3600"},  # Cache for 1 hour
        )


@dataclass
class ProtectedResourceMetadataHandler:
    metadata: ProtectedResourceMetadata

    async def handle(self, request: Request) -> Response:
        return PydanticJSONResponse(
            content=self.metadata,
            headers={"Cache-Control": "public, max-age=3600"},  # Cache for 1 hour
        )
