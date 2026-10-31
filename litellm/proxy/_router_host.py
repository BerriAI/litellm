"""
The proxy's RouterHost: answers the router's questions about pass-through credentials, member
auto-router access and auto-router prompt compression with the proxy's own modules.

Installed by litellm.router_host the first time the router asks for its host. Each method looks its
target up at call time, so patches of the proxy functions apply.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm.router_host import CompressionPolicy, RouterHost

if TYPE_CHECKING:
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.router import Router
    from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES


class ProxyRouterHost(RouterHost):
    def add_vertex_pass_through_credentials(
        self,
        project_id: str,
        location: str,
        vertex_credentials: "VERTEX_CREDENTIALS_TYPES | None",
    ) -> None:
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import passthrough_endpoint_router

        passthrough_endpoint_router.add_vertex_credentials(
            project_id=project_id,
            location=location,
            vertex_credentials=vertex_credentials,
        )

    async def authorize_strategy_inference(
        self,
        deployment: Mapping[str, object] | None,
        request_kwargs: Mapping[str, object],
        llm_router: "Router",
    ) -> None:
        from litellm.proxy.auth import auto_router_checks

        await auto_router_checks.authorize_member_auto_router_inference(
            deployment=deployment,
            request_kwargs=request_kwargs,
            llm_router=llm_router,
        )

    async def finalize_strategy_request(self, request_kwargs: Mapping[str, object]) -> None:
        from litellm.litellm_core_utils.core_helpers import budget_reservation_from_metadata
        from litellm.proxy.spend_tracking.budget_reservation import release_or_invalidate_budget_reservation

        metadata_values: Final = tuple(request_kwargs.get(key) for key in ("metadata", "litellm_metadata"))
        for metadata in metadata_values:
            if isinstance(metadata, Mapping):
                await release_or_invalidate_budget_reservation(
                    budget_reservation_from_metadata(TypeAdapter(Mapping[str, object]).validate_python(metadata))
                )

    async def bill_strategy_stream_disconnect(self, response: "CustomStreamWrapper") -> None:
        from litellm.proxy.common_request_processing import settle_strategy_stream_disconnect

        await settle_strategy_stream_disconnect(response)

    def compression_policy(
        self,
        llm_router: "Router | None",
        model_alias: str,
        request_kwargs: Mapping[str, object],
        request_tags: Sequence[str],
    ) -> CompressionPolicy | None:
        from litellm.proxy.guardrails import auto_router_compression

        return auto_router_compression.policy_for_model(
            llm_router=llm_router,
            model_alias=model_alias,
            request_kwargs=request_kwargs,
            request_tags=request_tags,
        )

    def model_hop_compression_armed(self) -> bool:
        from litellm.proxy.guardrails import auto_router_compression

        return auto_router_compression.model_hop_compression_armed()

    async def messages_for_routing(
        self,
        policy: CompressionPolicy | None,
        messages: list[dict[str, object]] | None,  # mutable-ok: shape fixed by the pre-routing hook protocol
        request_kwargs: Mapping[str, object],
    ) -> list[dict[str, object]] | None:  # mutable-ok: shape fixed by the pre-routing hook protocol
        from litellm.proxy.guardrails import auto_router_compression

        return await auto_router_compression.messages_for_routing(
            policy=policy,  # pyright: ignore[reportArgumentType]  # the proxy's own policy object, from compression_policy
            messages=messages,
            request_kwargs=request_kwargs,
        )
