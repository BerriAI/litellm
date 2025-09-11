import secrets
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, RootModel, ValidationError
from starlette.requests import Request
from starlette.responses import Response

from mcp.server.auth.errors import stringify_pydantic_error
from mcp.server.auth.json_response import PydanticJSONResponse
from mcp.server.auth.provider import (
    OAuthAuthorizationServerProvider,
    RegistrationError,
    RegistrationErrorCode,
)
from mcp.server.auth.settings import ClientRegistrationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata


class RegistrationRequest(RootModel[OAuthClientMetadata]):
    # this wrapper is a no-op; it's just to separate out the types exposed to the
    # provider from what we use in the HTTP handler
    root: OAuthClientMetadata


class RegistrationErrorResponse(BaseModel):
    error: RegistrationErrorCode
    error_description: str | None


@dataclass
class RegistrationHandler:
    provider: OAuthAuthorizationServerProvider[Any, Any, Any]
    options: ClientRegistrationOptions

    async def handle(self, request: Request) -> Response:
        # Implements dynamic client registration as defined in https://datatracker.ietf.org/doc/html/rfc7591#section-3.1
        try:
            # Parse request body as JSON
            body = await request.json()
            client_metadata = OAuthClientMetadata.model_validate(body)

            # Scope validation is handled below
        except ValidationError as validation_error:
            return PydanticJSONResponse(
                content=RegistrationErrorResponse(
                    error="invalid_client_metadata",
                    error_description=stringify_pydantic_error(validation_error),
                ),
                status_code=400,
            )

        client_id = str(uuid4())
        client_secret = None
        if client_metadata.token_endpoint_auth_method != "none":
            # cryptographically secure random 32-byte hex string
            client_secret = secrets.token_hex(32)

        if client_metadata.scope is None and self.options.default_scopes is not None:
            client_metadata.scope = " ".join(self.options.default_scopes)
        elif (
            client_metadata.scope is not None and self.options.valid_scopes is not None
        ):
            requested_scopes = set(client_metadata.scope.split())
            valid_scopes = set(self.options.valid_scopes)
            if not requested_scopes.issubset(valid_scopes):
                return PydanticJSONResponse(
                    content=RegistrationErrorResponse(
                        error="invalid_client_metadata",
                        error_description="Requested scopes are not valid: "
                        f"{', '.join(requested_scopes - valid_scopes)}",
                    ),
                    status_code=400,
                )
        if set(client_metadata.grant_types) != {"authorization_code", "refresh_token"}:
            return PydanticJSONResponse(
                content=RegistrationErrorResponse(
                    error="invalid_client_metadata",
                    error_description="grant_types must be authorization_code "
                    "and refresh_token",
                ),
                status_code=400,
            )

        client_id_issued_at = int(time.time())
        client_secret_expires_at = (
            client_id_issued_at + self.options.client_secret_expiry_seconds
            if self.options.client_secret_expiry_seconds is not None
            else None
        )

        client_info = OAuthClientInformationFull(
            client_id=client_id,
            client_id_issued_at=client_id_issued_at,
            client_secret=client_secret,
            client_secret_expires_at=client_secret_expires_at,
            # passthrough information from the client request
            redirect_uris=client_metadata.redirect_uris,
            token_endpoint_auth_method=client_metadata.token_endpoint_auth_method,
            grant_types=client_metadata.grant_types,
            response_types=client_metadata.response_types,
            client_name=client_metadata.client_name,
            client_uri=client_metadata.client_uri,
            logo_uri=client_metadata.logo_uri,
            scope=client_metadata.scope,
            contacts=client_metadata.contacts,
            tos_uri=client_metadata.tos_uri,
            policy_uri=client_metadata.policy_uri,
            jwks_uri=client_metadata.jwks_uri,
            jwks=client_metadata.jwks,
            software_id=client_metadata.software_id,
            software_version=client_metadata.software_version,
        )
        try:
            # Register client
            await self.provider.register_client(client_info)

            # Return client information
            return PydanticJSONResponse(content=client_info, status_code=201)
        except RegistrationError as e:
            # Handle registration errors as defined in RFC 7591 Section 3.2.2
            return PydanticJSONResponse(
                content=RegistrationErrorResponse(
                    error=e.error, error_description=e.error_description
                ),
                status_code=400,
            )
