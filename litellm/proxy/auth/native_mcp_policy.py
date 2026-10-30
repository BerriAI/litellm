from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter, ValidationError

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


@dataclass(frozen=True, slots=True)
class NativeMCPRequestContext:
    attempt_id: str
    accesses: tuple[NativeMCPAccess, ...]


_REQUEST_CONTEXT: Final = TypeAdapter(NativeMCPRequestContext)
_CONTEXT_FIELDS: Final = TypeAdapter(dict[str, object])
_CONTEXT_KEY: Final = "native_mcp_request_context"


def native_mcp_request_context(response: object) -> NativeMCPRequestContext | None:
    from litellm.litellm_core_utils.hidden_params import get_hidden_params

    hidden: Final = get_hidden_params(response)
    raw: Final = hidden.get(_CONTEXT_KEY) if hidden is not None else None
    if raw is None:
        return None
    return _REQUEST_CONTEXT.validate_python(raw)


def native_mcp_deployment_validator(
    identity: NativeMCPIdentity,
) -> Callable[[Mapping[str, object]], Callable[[object], object]] | None:
    policy: Final = resolve_native_mcp_policy(identity)
    if policy is None:
        return None

    def validate(data: Mapping[str, object]) -> Callable[[object], object]:
        from litellm.proxy.guardrails.tool_name_extraction import extract_request_mcp_accesses

        accesses: Final = extract_request_mcp_accesses(data)
        raise_native_mcp_denial(evaluate_native_mcp_policy(policy, accesses))
        assert not isinstance(accesses, NativeMCPDenial)
        context: Final = NativeMCPRequestContext(str(uuid4()), accesses)

        def bind(response: object) -> object:
            from litellm.litellm_core_utils.hidden_params import set_hidden_param
            from litellm.router_utils.add_retry_fallback_headers import prepare_response_for_header_attachment

            prepared: Final = prepare_response_for_header_attachment(response)
            if prepared is not None:
                set_hidden_param(
                    prepared,
                    _CONTEXT_KEY,
                    _CONTEXT_FIELDS.validate_python(_REQUEST_CONTEXT.dump_python(context, mode="json")),
                )
            return prepared

        return bind

    return validate


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


def raise_native_mcp_denial(denial: NativeMCPDenial | None) -> None:
    if denial is None:
        return
    from litellm.proxy._types import ProxyErrorTypes, ProxyException

    raise ProxyException(
        message=denial.message,
        type=ProxyErrorTypes.tool_access_denied,
        param="tools",
        code=400,
    )


def check_native_mcp_request(identity: NativeMCPIdentity, data: Mapping[str, object]) -> None:
    policy: Final = resolve_native_mcp_policy(identity)
    if policy is None:
        return
    from litellm.proxy.guardrails.tool_name_extraction import extract_request_mcp_accesses

    raise_native_mcp_denial(evaluate_native_mcp_policy(policy, extract_request_mcp_accesses(data)))


def check_native_mcp_response(
    identity: NativeMCPIdentity,
    data: Mapping[str, object],
    response: object,
    *,
    is_retrieval: bool = False,
) -> None:
    policy: Final = resolve_native_mcp_policy(identity)
    if policy is None:
        return
    from litellm.proxy.guardrails.tool_name_extraction import (
        extract_request_mcp_accesses,
        extract_response_mcp_accesses,
    )

    # Only retrieval binds from the provider's stored attachment list. Creation
    # must remain bound to the request validated before dispatch.
    response_fields: Final = as_str_mapping(response.model_dump() if isinstance(response, BaseModel) else response)
    binding: Final = {"tools": response_fields.get("tools")} if is_retrieval and response_fields is not None else data
    request_accesses: Final = extract_request_mcp_accesses(binding)
    raise_native_mcp_denial(evaluate_native_mcp_policy(policy, request_accesses))
    if isinstance(request_accesses, NativeMCPDenial) or is_gateway_only(request_accesses):
        return
    raise_native_mcp_denial(
        evaluate_native_mcp_policy(policy, extract_response_mcp_accesses(response, request_accesses))
    )
