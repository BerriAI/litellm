"""
RouterHost: what the router asks of the application it runs in.

The router used to import three proxy modules on demand (vertex pass-through credentials, member
auto-router authorization and auto-router prompt compression), so litellm.router and litellm.proxy
imported each other. The router now asks the installed host instead. The first time it asks, the
proxy's host (litellm/proxy/_router_host.py) is installed when the proxy can be imported,
whichever modules were imported before, so router.py itself never imports the proxy.

When the proxy can't be imported, the default host keeps no pass-through credentials, applies no
prompt compression, and refuses member auto-routers, which need the proxy's team checks.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol

import httpx

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES


class CompressionPolicy(Protocol):
    """An auto router's prompt-compression choice for the routing hop and the model hop."""

    @property
    def routing(self) -> str | None: ...

    @property
    def model(self) -> str | None: ...

    @property
    def is_same(self) -> bool: ...


class RouterHost:
    """The default host: no proxy. The proxy replaces it with ``install_router_host``."""

    def add_vertex_pass_through_credentials(
        self,
        project_id: str,
        location: str,
        vertex_credentials: "VERTEX_CREDENTIALS_TYPES | None",
    ) -> None:
        """Keep vertex credentials for pass-through endpoints. Only the proxy serves those."""

    async def authorize_strategy_inference(
        self,
        deployment: Mapping[str, object] | None,
        request_kwargs: Mapping[str, object],
        llm_router: "Router",
    ) -> None:
        """Raise when the caller may not use the routing strategy behind ``deployment``."""
        model_info: Final = deployment.get("model_info") if deployment is not None else None
        if isinstance(model_info, Mapping) and model_info.get("member_auto_router") is True:
            from litellm.exceptions import PermissionDeniedError

            raise PermissionDeniedError(
                message="Member auto-routers require the LiteLLM proxy's team access checks",
                llm_provider="",
                model=str(deployment.get("model_name", "")) if deployment is not None else "",
                response=httpx.Response(status_code=403, request=httpx.Request("POST", "https://litellm.invalid")),
            )

    def compression_policy(
        self,
        llm_router: "Router | None",
        model_alias: str,
        request_kwargs: Mapping[str, object],
        request_tags: Sequence[str],
    ) -> CompressionPolicy | None:
        """The prompt-compression policy of the auto router ``model_alias`` resolves to, if any."""
        return None

    def model_hop_compression_armed(self) -> bool:
        """Whether this request's model-hop compression already ran before routing."""
        return False

    async def messages_for_routing(
        self,
        policy: CompressionPolicy | None,
        messages: list[dict[str, object]] | None,  # mutable-ok: shape fixed by the pre-routing hook protocol
        request_kwargs: Mapping[str, object],
    ) -> list[dict[str, object]] | None:  # mutable-ok: shape fixed by the pre-routing hook protocol
        """Messages to make the routing decision on. None means route on the request's own messages."""
        return None


# The installed host, in a one-slot list so installing it needs no ``global``. None until first asked.
_installed: Final[list[RouterHost | None]] = [None]  # mutable-ok: one slot, replaced by install_router_host


def install_router_host(host: RouterHost) -> None:
    """Make ``host`` answer the router's questions from now on (the proxy's host module calls this)."""
    _installed[0] = host


def router_host() -> RouterHost:
    """The installed host. Nothing installed: the proxy's host if the proxy imports, else the default."""
    if _installed[0] is None:
        try:
            from litellm.proxy._router_host import ProxyRouterHost
        except ImportError:
            _installed[0] = RouterHost()
        else:
            _installed[0] = ProxyRouterHost()
    return _installed[0]
