from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

from litellm.proxy.agent_endpoints.managed_identity import classify_agent_subject
from litellm.repositories.table_repositories import (
    AgentIdentityRepository,
    AgentsRepository,
    RetiredAgentIdentityRepository,
    RetiredAgentRepository,
    VerifiedSubjectRepository,
)
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import (
    AgentIdentityFailure,
    ManagedAgentContext,
    MicrosoftInteractiveSubject,
    VerifiedHumanSubject,
)

if TYPE_CHECKING:
    from prisma.models import LiteLLM_VerifiedSubject
    from prisma.types import (
        LiteLLM_AgentIdentityUpdateManyMutationInput,
        LiteLLM_AgentIdentityWhereInput,
        LiteLLM_AgentIdentityWhereUniqueInput,
        LiteLLM_AgentsTableInclude,
        LiteLLM_AgentsTableWhereUniqueInput,
        LiteLLM_VerifiedSubjectCreateInput,
        LiteLLM_VerifiedSubjectUpsertInput,
        LiteLLM_VerifiedSubjectWhereUniqueInput,
    )


class AgentIdentityStore:
    @classmethod
    def from_client(cls, client: object) -> "AgentIdentityStore":
        return cls(
            AgentsRepository(client, use_writer=True),
            AgentIdentityRepository(client, use_writer=True),
            VerifiedSubjectRepository(client, use_writer=True),
            RetiredAgentIdentityRepository(client, use_writer=True),
            RetiredAgentRepository(client, use_writer=True),
        )

    def __init__(
        self,
        agents: AgentsRepository,
        identities: AgentIdentityRepository,
        humans: VerifiedSubjectRepository,
        retired: RetiredAgentIdentityRepository | None = None,
        retired_agents: RetiredAgentRepository | None = None,
    ) -> None:
        self.agents = agents
        self.identities = identities
        self.humans = humans
        self.retired = retired
        self.retired_agents = retired_agents

    async def agent(self, agent_id: str) -> AgentResponse | AgentIdentityFailure | None:
        try:
            where: Final[LiteLLM_AgentsTableWhereUniqueInput] = {"agent_id": agent_id}
            include: Final[LiteLLM_AgentsTableInclude] = {
                "identity": True,
                "object_permission": True,
                "litellm_budget_table": True,
            }
            row: Final = await self.agents.table.find_unique(where=where, include=include)
            if row is None:
                return None
            return AgentResponse.model_validate(row.model_dump())
        except Exception:
            return AgentIdentityFailure(code="policy_unavailable", message="Agent policy could not be loaded")

    async def unbound_client(self, where: "LiteLLM_AgentIdentityWhereUniqueInput") -> AgentIdentityFailure | None:
        if self.retired is not None:
            try:
                retired: Final = await self.retired.table.find_unique(where=where)
            except Exception:
                return AgentIdentityFailure(
                    code="policy_unavailable", message="Retired agent identity could not be checked"
                )
            if retired is not None:
                return AgentIdentityFailure(message="This agent identity binding has been retired")
        return None

    async def resolve_verified_claims(
        self, claims: Mapping[str, object]
    ) -> ManagedAgentContext | AgentIdentityFailure | None:
        issuer: Final = claims.get("iss")
        tenant: Final = claims.get("tid")
        client: Final = claims.get("azp")
        if not isinstance(issuer, str) or not isinstance(tenant, str) or not isinstance(client, str):
            return None
        where: Final[LiteLLM_AgentIdentityWhereUniqueInput] = {
            "provider_tenant_id_client_id": {"provider": "microsoft_entra", "tenant_id": tenant, "client_id": client}
        }
        try:
            row: Final = await self.identities.table.find_unique(where=where)
        except Exception:
            return AgentIdentityFailure(code="policy_unavailable", message="Agent identity could not be loaded")
        proven: Final = await self.subject(issuer, tenant, claims.get("oid"))
        if isinstance(proven, AgentIdentityFailure):
            return proven
        if row is None:
            return await self.unbound_client(where)
        agent: Final = await self.agent(row.agent_id)
        if isinstance(agent, AgentIdentityFailure):
            return agent
        if (
            agent is None
            or not agent.identity_managed
            or not agent.enabled
            or agent.identity is None
            or not agent.identity.active
        ):
            return AgentIdentityFailure(message="Agent is disabled or no longer bound to an identity")
        subject: Final = classify_agent_subject(agent.identity, claims, agent.execution_mode)
        if isinstance(subject, AgentIdentityFailure):
            return subject
        if subject.kind == "application":
            return ManagedAgentContext(
                agent_id=agent.agent_id,
                binding_revision=agent.identity.revision,
                mode=subject.mode,
                subject_oid=subject.oid,
            )
        human: Final = (
            VerifiedHumanSubject.model_validate(proven.model_dump())
            if proven is not None
            and proven.kind == "human"
            and proven.verified_via == "sso_interactive"
            and proven.user_id is not None
            else None
        )
        if human is None:
            return AgentIdentityFailure(message="The delegated user must first sign in through trusted Microsoft SSO")
        return ManagedAgentContext(
            agent_id=agent.agent_id,
            binding_revision=agent.identity.revision,
            mode=subject.mode,
            user_id=human.user_id,
            subject_oid=subject.oid,
        )

    async def subject(
        self, issuer: str, tenant_id: str, oid: object
    ) -> "LiteLLM_VerifiedSubject | AgentIdentityFailure | None":
        if not isinstance(oid, str):
            return None
        try:
            where: Final[LiteLLM_VerifiedSubjectWhereUniqueInput] = {
                "issuer_tenant_id_oid": {"issuer": issuer, "tenant_id": tenant_id, "oid": oid}
            }
            return await self.humans.table.find_unique(where=where)
        except Exception:
            return AgentIdentityFailure(code="policy_unavailable", message="Subject classification is unavailable")

    async def retired_agent(self, agent_id: str) -> bool | AgentIdentityFailure:
        if self.retired_agents is None:
            return AgentIdentityFailure(code="policy_unavailable", message="Agent history is unavailable")
        try:
            return await self.retired_agents.table.find_unique(where={"original_agent_id": agent_id}) is not None
        except Exception:
            return AgentIdentityFailure(code="policy_unavailable", message="Agent history is unavailable")

    async def record_authentication(self, context: ManagedAgentContext) -> AgentIdentityFailure | None:
        try:
            if context.binding_revision is None:
                return AgentIdentityFailure(message="Agent authentication requires a binding revision")
            where: Final[LiteLLM_AgentIdentityWhereInput] = {
                "agent_id": context.agent_id,
                "revision": context.binding_revision,
            }
            data: Final[LiteLLM_AgentIdentityUpdateManyMutationInput] = {
                "last_authenticated_at": datetime.now(timezone.utc)
            }
            count: Final = await self.identities.table.update_many(where=where, data=data)
            if count != 1:
                return AgentIdentityFailure(message="Agent identity changed during authentication; retry")
            return None
        except Exception:
            return AgentIdentityFailure(code="policy_unavailable", message="Agent authentication could not be recorded")

    async def enroll_interactive_human(
        self,
        subject: MicrosoftInteractiveSubject,
        user_id: str,
    ) -> AgentIdentityFailure | None:
        try:
            where: Final[LiteLLM_VerifiedSubjectWhereUniqueInput] = {
                "issuer_tenant_id_oid": {"issuer": subject.issuer, "tenant_id": subject.tenant_id, "oid": subject.oid}
            }
            create_data: Final[LiteLLM_VerifiedSubjectCreateInput] = {
                "issuer": subject.issuer,
                "tenant_id": subject.tenant_id,
                "oid": subject.oid,
                "user_id": user_id,
                "verified_via": "sso_interactive",
            }
            data: Final[LiteLLM_VerifiedSubjectUpsertInput] = {"create": create_data, "update": {}}
            row: Final = await self.humans.table.upsert(where=where, data=data)
            if row.kind != "human" or row.user_id != user_id or row.verified_via != "sso_interactive":
                return AgentIdentityFailure(message="Microsoft subject is already bound to another local identity")
            return None
        except Exception:
            return AgentIdentityFailure(
                code="policy_unavailable", message="Microsoft subject enrollment is unavailable"
            )


async def resolve_managed_agent(
    claims: Mapping[str, object],
    client: object,
) -> ManagedAgentContext | None:
    from litellm.proxy.agent_endpoints.managed_identity import raise_identity_failure

    if client is None:
        return None
    result: Final = await AgentIdentityStore.from_client(client).resolve_verified_claims(claims)
    if isinstance(result, AgentIdentityFailure):
        raise_identity_failure(result)
    return result
