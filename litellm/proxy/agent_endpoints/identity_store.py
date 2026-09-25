from collections.abc import Mapping
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import AGENT_IDENTITY_MISS_CACHE_MAX_ITEMS, AGENT_IDENTITY_MISS_CACHE_TTL
from litellm.proxy.agent_endpoints.managed_identity import classify_agent_subject
from litellm.repositories.table_repositories import (
    AgentIdentityRepository,
    AgentsRepository,
    RetiredAgentIdentityRepository,
    RetiredAgentRepository,
    SCIMResourceRepository,
    SCIMSourceRepository,
    VerifiedSubjectRepository,
)
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import (
    AgentIdentityFailure,
    ManagedAgentContext,
    MicrosoftInteractiveSubject,
    VerifiedAgentSubject,
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

UNBOUND_CLAIMS: Final = InMemoryCache(
    max_size_in_memory=AGENT_IDENTITY_MISS_CACHE_MAX_ITEMS, default_ttl=AGENT_IDENTITY_MISS_CACHE_TTL
)


def forget_unbound_claims() -> None:
    UNBOUND_CLAIMS.flush_cache()


class AgentIdentityStore:
    @classmethod
    def from_client(cls, client: object) -> "AgentIdentityStore":
        return cls(
            AgentsRepository(client, use_writer=True),
            AgentIdentityRepository(client, use_writer=True),
            VerifiedSubjectRepository(client, use_writer=True),
            RetiredAgentIdentityRepository(client, use_writer=True),
            SCIMSourceRepository(client, use_writer=True),
            SCIMResourceRepository(client, use_writer=True),
            RetiredAgentRepository(client, use_writer=True),
            UNBOUND_CLAIMS,
        )

    def __init__(
        self,
        agents: AgentsRepository,
        identities: AgentIdentityRepository,
        humans: VerifiedSubjectRepository,
        retired: RetiredAgentIdentityRepository | None = None,
        sources: SCIMSourceRepository | None = None,
        resources: SCIMResourceRepository | None = None,
        retired_agents: RetiredAgentRepository | None = None,
        unbound: InMemoryCache | None = None,
    ) -> None:
        self.agents = agents
        self.identities = identities
        self.humans = humans
        self.retired = retired
        self.sources = sources
        self.resources = resources
        self.retired_agents = retired_agents
        self.unbound = unbound

    async def agent(self, agent_id: str) -> AgentResponse | AgentIdentityFailure | None:
        try:
            where: Final[LiteLLM_AgentsTableWhereUniqueInput] = {"agent_id": agent_id}
            include: Final[LiteLLM_AgentsTableInclude] = {
                "identity": True,
                "object_permission": True,
            }
            row: Final = await self.agents.table.find_unique(where=where, include=include)
            if row is None:
                return None
            agent: Final = AgentResponse.model_validate(row.model_dump())
            if agent.identity is None or agent.identity.provisioning_source_id is None:
                return agent
            return await self.directory_policy(agent)
        except Exception:
            return AgentIdentityFailure(code="policy_unavailable", message="Agent policy could not be loaded")

    async def unbound_client(
        self, where: "LiteLLM_AgentIdentityWhereUniqueInput", miss_key: str
    ) -> AgentIdentityFailure | None:
        if self.retired is not None:
            try:
                retired: Final = await self.retired.table.find_unique(where=where)
            except Exception:
                return AgentIdentityFailure(
                    code="policy_unavailable", message="Retired agent identity could not be checked"
                )
            if retired is not None:
                return AgentIdentityFailure(message="This agent identity binding has been retired")
        if self.unbound is not None:
            self.unbound.set_cache(miss_key, True)
        return None

    async def resolve_verified_claims(
        self, claims: Mapping[str, object]
    ) -> ManagedAgentContext | AgentIdentityFailure | None:
        issuer: Final = claims.get("iss")
        tenant: Final = claims.get("tid")
        client: Final = claims.get("azp")
        if not isinstance(issuer, str) or not isinstance(tenant, str) or not isinstance(client, str):
            return None
        miss_key: Final = f"agent-identity-miss:{issuer}|{tenant}|{client}|{claims.get('oid')}"
        if self.unbound is not None and self.unbound.get_cache(miss_key) is not None:
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
        if (
            proven is not None
            and proven.kind == "agent_user"
            and (row is None or proven.agent_id != row.agent_id or proven.parent_client_id != client)
        ):
            return AgentIdentityFailure(message="Provisioned subject does not match an active agent binding")
        if row is None:
            return await self.unbound_client(where, miss_key)
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
        native: Final = (
            VerifiedAgentSubject.model_validate(proven.model_dump())
            if proven is not None and proven.kind == "agent_user" and proven.agent_id is not None
            else None
        )
        if native is not None and (
            agent.identity.provisioning_source_id is None
            or not agent.directory_active
            or agent.directory_access_group_ids == ()
        ):
            return AgentIdentityFailure(message="Provisioned agent is inactive or has no mapped directory entitlement")
        subject: Final = classify_agent_subject(agent.identity, claims, agent.execution_mode, native_subject=native)
        if isinstance(subject, AgentIdentityFailure):
            return subject
        if subject.kind in ("application", "agent_user"):
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

    async def directory_policy(self, agent: AgentResponse) -> AgentResponse | AgentIdentityFailure:
        from pydantic import TypeAdapter

        from litellm.types.proxy.management_endpoints.scim_agent_provisioning import SCIMGroupMapping

        if self.sources is None or self.resources is None or agent.identity is None:
            return AgentIdentityFailure(code="policy_unavailable", message="Provisioning state is unavailable")
        from prisma.types import LiteLLM_SCIMResourceWhereInput, LiteLLM_SCIMSourceWhereUniqueInput

        source_where: Final[LiteLLM_SCIMSourceWhereUniqueInput] = {"source_id": agent.identity.provisioning_source_id}
        resource_where: Final[LiteLLM_SCIMResourceWhereInput] = {
            "source_id": agent.identity.provisioning_source_id,
            "kind": "Users",
            "local_id": agent.agent_id,
        }
        source: Final = await self.sources.table.find_unique(where=source_where)
        subjects: Final = await self.resources.table.find_many(where=resource_where)
        live: Final = tuple(subject for subject in subjects if subject.active and not subject.deleted)
        if source is None or not source.enabled or len(live) != 1:
            return agent.model_copy(
                update=MappingProxyType({"directory_active": False, "directory_access_group_ids": ()})
            )
        from litellm.proxy.management_endpoints.scim.agent_provisioning import user_document

        directory_user: Final = user_document(live[0])
        if (
            source.tenant_id != agent.identity.tenant_id
            or directory_user.agent_user is None
            or str(directory_user.agent_user.identityParentId) != agent.identity.client_id
        ):
            return AgentIdentityFailure(message="Directory identity does not match the registered binding")
        proven: Final = await self.subject(agent.identity.issuer, source.tenant_id, live[0].external_id)
        if isinstance(proven, AgentIdentityFailure):
            return proven
        if (
            proven is None
            or proven.kind != "agent_user"
            or proven.verified_via != "scim"
            or proven.agent_id != agent.agent_id
            or proven.parent_client_id != agent.identity.client_id
            or proven.scim_resource_id != live[0].id
        ):
            return AgentIdentityFailure(message="Directory subject does not match the provisioned resource")
        groups_where: Final[LiteLLM_SCIMResourceWhereInput] = {
            "source_id": source.source_id,
            "kind": "Groups",
            "deleted": False,
            "active": True,
            "member_ids": {"has": live[0].id},
        }
        groups: Final = await self.resources.table.find_many(where=groups_where)
        mappings: Final = TypeAdapter(tuple[SCIMGroupMapping, ...]).validate_python(source.group_mappings)
        external_ids: Final = frozenset(group.external_id for group in groups)
        mapped: Final = tuple(
            mapping.access_group_ids for mapping in mappings if str(mapping.external_group_id) in external_ids
        )
        ids: Final = tuple(sorted(frozenset(chain.from_iterable(mapped))))
        return agent.model_copy(update=MappingProxyType({"directory_active": True, "directory_access_group_ids": ids}))

    async def retired_agent(self, agent_id: str) -> bool | AgentIdentityFailure:
        if self.retired_agents is None:
            return AgentIdentityFailure(code="policy_unavailable", message="Agent history is unavailable")
        try:
            return await self.retired_agents.table.find_unique(where={"original_agent_id": agent_id}) is not None
        except Exception:
            return AgentIdentityFailure(code="policy_unavailable", message="Agent history is unavailable")

    async def verified_human(
        self, issuer: str, tenant_id: str, oid: str
    ) -> VerifiedHumanSubject | AgentIdentityFailure | None:
        try:
            where: Final[LiteLLM_VerifiedSubjectWhereUniqueInput] = {
                "issuer_tenant_id_oid": {"issuer": issuer, "tenant_id": tenant_id, "oid": oid}
            }
            row: Final = await self.humans.table.find_unique(where=where)
            if row is None or row.kind != "human" or row.verified_via != "sso_interactive":
                return None
            return VerifiedHumanSubject.model_validate(row.model_dump())
        except Exception:
            return AgentIdentityFailure(code="policy_unavailable", message="Delegated subject could not be verified")

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
