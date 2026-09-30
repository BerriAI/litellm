from typing import Final

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.identity_store import AgentIdentityStore
from litellm.proxy.agent_endpoints.managed_identity import raise_identity_failure
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import AgentIdentityFailure, ManagedAgentContext


def managed_agent_policy(auth: "UserAPIKeyAuth | None") -> AgentResponse | None:
    """The admitted managed policy, or ``None`` when the subject was never admitted as a managed agent.

    ``admit_managed_actor`` only assigns ``managed_agent_policy`` after ``actor_admission_failure``
    has verified the bound context, so an ``AgentResponse`` here means admission succeeded.
    """
    policy: Final = auth.managed_agent_policy if auth is not None else None
    return policy if isinstance(policy, AgentResponse) else None


async def admit_managed_actor(auth: UserAPIKeyAuth, store: AgentIdentityStore | None) -> None:
    delegation_verified: Final = auth._managed_delegation_verified  # pyright: ignore[reportPrivateUsage]  # the one-shot delegation marker is a PrivateAttr by design
    auth._managed_delegation_verified = False  # pyright: ignore[reportPrivateUsage]  # consumed here so a replayed token cannot reuse it
    if auth.agent_id is None:
        return
    if store is None:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        registered: Final = global_agent_registry.get_agent_by_id(auth.agent_id)
        if auth.managed_agent_context is not None or (
            registered is not None and (registered.identity_managed or registered.identity is not None)
        ):
            raise_identity_failure(
                AgentIdentityFailure(code="policy_unavailable", message="Managed agent policy requires a database")
            )
        return
    agent: Final = await store.agent(auth.agent_id)
    if isinstance(agent, AgentIdentityFailure):
        raise_identity_failure(agent)
    if agent is None:
        retired: Final = await store.retired_agent(auth.agent_id)
        if isinstance(retired, AgentIdentityFailure):
            raise_identity_failure(retired)
        if auth.managed_agent_context is not None or retired:
            raise_identity_failure(AgentIdentityFailure(message="Agent no longer exists"))
        return
    if not agent.identity_managed:
        return
    if auth.jwt_claims and auth.managed_agent_context is None:
        raise_identity_failure(AgentIdentityFailure(message="A managed agent requires a matching verified identity"))
    failure: Final = actor_admission_failure(agent, auth.managed_agent_context)
    if failure is not None:
        raise_identity_failure(failure)
    auth.managed_agent_policy = agent
    auth.billing_agent_policy = agent
    auth.requires_fresh_policy = True
    if (
        auth.managed_agent_context is not None
        and auth.managed_agent_context.mode == "delegated"
        and not delegation_verified
    ):
        from litellm.proxy.agent_endpoints.auth.agent_permission_handler import verified_human_agent_grants

        grants: Final = await verified_human_agent_grants(auth.managed_agent_context.user_id, auth.team_id)
        if agent.agent_id not in grants:
            raise_identity_failure(
                AgentIdentityFailure(message="The delegated user is not permitted to invoke this agent")
            )


def actor_admission_failure(
    agent: AgentResponse,
    context: ManagedAgentContext | None,
) -> AgentIdentityFailure | None:
    if not agent.enabled or agent.identity is None or not agent.identity.active:
        return AgentIdentityFailure(message="Agent execution is disabled")
    if context is None:
        return AgentIdentityFailure(message="This agent requires its bound identity provider token")
    if context.agent_id != agent.agent_id or context.binding_revision != agent.identity.revision:
        return AgentIdentityFailure(message="Agent identity changed during authentication; retry")
    if agent.execution_mode not in (context.mode, "both"):
        return AgentIdentityFailure(message="Agent is not enabled for this execution mode")
    if context.mode == "delegated" and not context.user_id:
        return AgentIdentityFailure(message="A verified human subject is required")
    return None
