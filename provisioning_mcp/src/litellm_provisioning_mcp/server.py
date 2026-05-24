"""FastMCP server exposing LiteLLM provisioning tools over streamable HTTP.

Authentication: the server is an OAuth 2.0 resource server. Every request must
carry a ``Bearer`` access token that validates against the configured issuer's
JWKS and carries the required scope. Tokens are never issued here.
"""

from __future__ import annotations

import logging

import anyio
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP

from .auth import JWKSValidator, TokenValidationError
from .commands import CommandError, CommandTimeout
from .config import Settings
from .helm import HelmRunner
from .kubectl import KubectlRunner
from .provisioner import Provisioner, ProvisionError, ProvisionRequest

logger = logging.getLogger("litellm_provisioning_mcp")


class JWTTokenVerifier:
    """Adapts :class:`JWKSValidator` to the MCP ``TokenVerifier`` protocol."""

    def __init__(self, validator: JWKSValidator, *, resource: str) -> None:
        self._validator = validator
        self._resource = resource

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            verified = await anyio.to_thread.run_sync(self._validator.validate, token)
        except TokenValidationError as exc:
            logger.warning("rejected bearer token: %s", exc)
            return None
        return AccessToken(
            token=token,
            client_id=verified.client_id,
            scopes=verified.scopes,
            expires_at=verified.expires_at,
            resource=self._resource,
        )


def _error(exc: Exception) -> dict:
    if isinstance(exc, CommandError):
        return {
            "success": False,
            "error": "command_failed",
            "returncode": exc.result.returncode,
            "detail": exc.result.stderr.strip() or exc.result.stdout.strip(),
        }
    if isinstance(exc, CommandTimeout):
        return {"success": False, "error": "timeout", "detail": str(exc)}
    if isinstance(exc, ProvisionError):
        return {"success": False, "error": "invalid_request", "detail": str(exc)}
    raise exc


def build_server(settings: Settings) -> FastMCP:
    validator = JWKSValidator(
        jwks_url=settings.oauth_jwks_url,
        issuer=settings.oauth_issuer,
        audience=settings.oauth_audience,
        algorithms=settings.oauth_algorithms,
        required_scope=settings.oauth_required_scope,
    )
    provisioner = Provisioner(
        settings,
        helm=HelmRunner(
            namespace=settings.namespace,
            binary=settings.helm_binary,
            wait_timeout=settings.command_timeout,
        ),
        kubectl=KubectlRunner(
            namespace=settings.namespace, binary=settings.kubectl_binary
        ),
    )

    mcp = FastMCP(
        name="litellm-provisioning",
        instructions=(
            "Provision and tear down ephemeral LiteLLM deployments for end-to-end "
            "testing. Provide the litellm repo URL and the git revision whose images "
            "should run; optionally enable a throwaway Postgres and/or Redis."
        ),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        stateless_http=True,
        json_response=True,
        token_verifier=JWTTokenVerifier(
            validator, resource=settings.resource_server_url
        ),
        auth=AuthSettings(
            issuer_url=settings.oauth_issuer,
            resource_server_url=settings.resource_server_url,
            required_scopes=[settings.oauth_required_scope],
        ),
    )

    @mcp.tool()
    async def provision_litellm_deployment(
        repo_url: str,
        revision: str,
        release_name: str | None = None,
        enable_postgres: bool = True,
        enable_redis: bool = False,
        enable_ui: bool = False,
        service_account: str | None = None,
        image_registry: str | None = None,
        external_database: dict | None = None,
        extra_values: dict | None = None,
    ) -> dict:
        """Provision (or upgrade) a LiteLLM deployment in the target namespace.

        Args:
            repo_url: Git URL of the litellm repository under test (used to derive
                the image registry, e.g. a fork's ``ghcr.io/<owner>``).
            revision: Git revision; used as the container image tag for every
                component, so an image with this tag must already be published.
            release_name: Optional helm release name (defaults to a sanitized
                ``<prefix>-<revision>``). Re-using a name upgrades in place.
            enable_postgres: Stand up a throwaway in-cluster Postgres (default).
            enable_redis: Stand up a throwaway in-cluster Redis.
            enable_ui: Also deploy the dashboard UI component.
            service_account: Existing ServiceAccount the litellm pods should run as.
            image_registry: Override the derived image registry base.
            external_database: Use an existing DB instead of an ephemeral one. Keys:
                host, dbname, secret_name (required); port, username_key, password_key.
            extra_values: Helm values deep-merged last (escape hatch).
        """
        request = ProvisionRequest(
            repo_url=repo_url,
            revision=revision,
            release_name=release_name,
            enable_postgres=enable_postgres,
            enable_redis=enable_redis,
            enable_ui=enable_ui,
            service_account=service_account,
            image_registry=image_registry,
            external_database=external_database,
            extra_values=extra_values or {},
        )
        try:
            return await provisioner.provision(request)
        except (ProvisionError, CommandError, CommandTimeout) as exc:
            return _error(exc)

    @mcp.tool()
    async def delete_litellm_deployment(release_name: str) -> dict:
        """Tear down a deployment: uninstall the helm release and its datastores."""
        try:
            return await provisioner.delete(release_name)
        except (ProvisionError, CommandError, CommandTimeout) as exc:
            return _error(exc)

    @mcp.tool()
    async def get_litellm_deployment_status(release_name: str) -> dict:
        """Report helm release status and pod readiness for a deployment."""
        try:
            return await provisioner.status(release_name)
        except (ProvisionError, CommandError, CommandTimeout) as exc:
            return _error(exc)

    @mcp.tool()
    async def list_litellm_deployments() -> dict:
        """List the helm releases in the target namespace."""
        try:
            return await provisioner.list_deployments()
        except (CommandError, CommandTimeout) as exc:
            return _error(exc)

    return mcp


def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    build_server(settings).run(transport="streamable-http")


if __name__ == "__main__":
    main()
