from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias

from pydantic import ValidationError

from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.proxy.native_mcp_policy import MCPToolScope, NativeMCPPolicy


class NativeMCPIdentity(Protocol):
    @property
    def metadata(self) -> object: ...

    @property
    def team_metadata(self) -> object: ...

    @property
    def request_route(self) -> str | None: ...


_NATIVE_MCP_CALL_TYPES: Final = frozenset(
    {"completion", "acompletion", "responses", "aresponses", "anthropic_messages"}
)


@dataclass(frozen=True, slots=True)
class NativeMCPAccess:
    server_name: str
    server_url: str
    tools: MCPToolScope
    source: Literal["remote", "gateway"] = "remote"


@dataclass(frozen=True, slots=True)
class NativeMCPDenial:
    message: str


NativeMCPExtraction: TypeAlias = tuple[NativeMCPAccess, ...] | NativeMCPDenial
NativeMCPResolution: TypeAlias = NativeMCPPolicy | NativeMCPDenial | None


def is_gateway_only(accesses: tuple[NativeMCPAccess, ...]) -> bool:
    return bool(accesses) and all(access.source == "gateway" for access in accesses)


def _metadata_policy(metadata: object) -> NativeMCPResolution:
    fields: Final = as_str_mapping(metadata)
    raw: Final = fields.get("native_mcp_policy") if fields is not None else None
    if raw is None:
        return None
    try:
        return NativeMCPPolicy.model_validate(raw)
    except ValidationError:
        return NativeMCPDenial("The key/team native MCP policy is invalid")


def resolve_native_mcp_policy(identity: NativeMCPIdentity) -> NativeMCPResolution:
    route: Final = identity.request_route
    if route is not None and not any(
        call_type.value in _NATIVE_MCP_CALL_TYPES for call_type in (get_call_types_for_route(route) or ())
    ):
        return None
    team: Final = _metadata_policy(identity.team_metadata)
    key: Final = _metadata_policy(identity.metadata)
    if isinstance(team, NativeMCPDenial):
        return team
    if isinstance(key, NativeMCPDenial):
        return key
    if team is None:
        return key
    if key is None:
        return team
    return NativeMCPPolicy(mcp_servers=(*team.mcp_servers, *key.mcp_servers))


def _check_access(policy: NativeMCPPolicy, access: NativeMCPAccess) -> NativeMCPDenial | None:
    if access.source == "gateway":
        return None
    grants: Final = tuple(
        grant for grant in policy.mcp_servers if grant.name == access.server_name and grant.url == access.server_url
    )
    if not grants:
        return NativeMCPDenial(f"MCP server '{access.server_name}' has no grant for the requested URL")
    if any(grant.tools == "*" for grant in grants):
        return None
    if access.tools == "*":
        return NativeMCPDenial(
            f"MCP server '{access.server_name}' requires a whole-server grant for an unbounded attachment"
        )
    allowed: Final = frozenset[str]().union(*(grant.tools for grant in grants))
    blocked: Final = tuple(tool for tool in access.tools if tool not in allowed)
    if blocked:
        return NativeMCPDenial(f"MCP server '{access.server_name}' blocks tool(s): {', '.join(blocked)}")
    return None


def evaluate_native_mcp_policy(policy: NativeMCPResolution, accesses: NativeMCPExtraction) -> NativeMCPDenial | None:
    if policy is None:
        return None
    if isinstance(policy, NativeMCPDenial):
        return policy
    if isinstance(accesses, NativeMCPDenial):
        return accesses
    if any(access.source == "gateway" for access in accesses) and not is_gateway_only(accesses):
        return NativeMCPDenial(
            "Mixing provider-native MCP and LiteLLM gateway attachments is not supported with native_mcp_policy"
        )
    failures: Final = tuple(_check_access(policy, access) for access in accesses)
    messages: Final = tuple(failure.message for failure in failures if failure is not None)
    return NativeMCPDenial("; ".join(messages)) if messages else None
