from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,  # noqa: TID251  # moved UserAPIKeyAuth fields and the Span alias are typed Any
    Final,
    TypeAlias,
)

from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from typing_extensions import TypeIs  # noqa: TID251  # narrows the before-validator input without copying it

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.team import Member
from litellm.models.verification_token import LiteLLM_VerificationToken
from litellm.types.agents import AgentCaller, AgentResponse
from litellm.types.proxy.agent_identity import ManagedAgentContext
from litellm.types.proxy.auth.user_roles import LitellmUserRoles
from litellm.types.proxy.carried_budget_state import (
    OrgBudgetSnapshot,
    TeamBudgetSnapshot,
    UserBudgetSnapshot,
)
from litellm.types.router import AllowedModelRegion

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span: TypeAlias = _Span | Any
else:
    Span: TypeAlias = Any


def is_jwt(token: str | None) -> bool:
    if token is None:
        return False
    parts: Final = token.split(".")
    return len(parts) == 3


def hash_token(token: str) -> str:
    import hashlib

    # Hash the string using SHA-256
    hashed_token: Final = hashlib.sha256(token.encode()).hexdigest()

    return hashed_token


_UntypedDict: TypeAlias = dict[Any, Any]


def _is_str_keyed_dict(value: object) -> TypeIs[dict[str, object]]:  # guard-ok: pydantic input and JSON keys are str
    return isinstance(value, dict)


class LiteLLM_VerificationTokenView(LiteLLM_VerificationToken):
    """
    Combined view of litellm verification token + litellm team table (select values)
    """

    team_spend: float | None = None
    team_alias: str | None = None
    team_tpm_limit: int | None = None
    team_rpm_limit: int | None = None
    team_tpd_limit: int | None = None
    team_max_budget: float | None = None
    team_soft_budget: float | None = None
    team_model_max_budget: dict[str, object] | None = None
    team_models: list[Any] = []
    team_blocked: bool = False
    soft_budget: float | None = None
    team_model_aliases: _UntypedDict | None = None
    team_member: Member | None = None
    team_metadata: _UntypedDict | None = None
    team_object_permission_id: str | None = None

    # Team Member Specific Params
    team_member_spend: float | None = None
    team_member_tpm_limit: int | None = None
    team_member_rpm_limit: int | None = None

    # End User Params
    end_user_id: str | None = None
    end_user_tpm_limit: int | None = None
    end_user_rpm_limit: int | None = None
    end_user_tpd_limit: int | None = None
    end_user_max_budget: float | None = None
    end_user_model_max_budget: _UntypedDict | None = None

    # Organization Params
    organization_alias: str | None = None
    organization_max_budget: float | None = None
    organization_tpm_limit: int | None = None
    organization_rpm_limit: int | None = None
    organization_metadata: _UntypedDict | None = None

    # Project Params
    project_alias: str | None = None
    project_metadata: _UntypedDict | None = None

    # Time stamps
    last_refreshed_at: float | None = None  # last time joint view was pulled from db

    def __init__(self, **kwargs: object) -> None:
        # Handle litellm_budget_table_* keys (budget table overrides when key value is None or empty)
        for key, value in list(kwargs.items()):
            if key.startswith("litellm_budget_table_") and value is not None:
                # Extract the corresponding attribute name
                attr_name = key.replace("litellm_budget_table_", "")
                # Use key's value from kwargs (from DB view), not class default
                current = kwargs.get(attr_name)
                if current is None:
                    current = getattr(self, attr_name, None)
                # Apply budget value when key has no value, or for model_max_budget when key has empty dict
                should_apply = current is None or (
                    attr_name == "model_max_budget" and _is_str_keyed_dict(current) and len(current) == 0
                )
                if should_apply:
                    kwargs[attr_name] = value
            if key == "end_user_id" and value is not None and isinstance(value, int):
                kwargs[key] = str(value)

        if kwargs.get("organization_id") is not None:
            kwargs["org_id"] = kwargs.pop("organization_id")
        # Initialize the superclass
        super().__init__(**kwargs)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]  # pydantic validates raw values


class UserAPIKeyAuth(LiteLLM_VerificationTokenView):  # the expected response object for user api key auth
    """
    Return the row in the db
    """

    api_key: str | None = None
    user_role: LitellmUserRoles | None = None
    allowed_model_region: AllowedModelRegion | None = None
    parent_otel_span: Span | None = None
    rpm_limit_per_model: dict[str, int] | None = None
    tpm_limit_per_model: dict[str, int] | None = None
    user_tpm_limit: int | None = None
    user_rpm_limit: int | None = None
    user_email: str | None = None
    user_spend: float | None = None
    user_max_budget: float | None = None
    # Values stay `object` rather than BudgetConfig: this is the raw JSON column,
    # and validating it here would make one malformed row fail auth outright.
    # resolve_model_budget validates the single entry a request actually needs.
    user_model_max_budget: Mapping[str, object] | None = None
    team_member_model_max_budget: Mapping[str, object] | None = Field(default=None, exclude=True)
    request_route: str | None = None
    is_session_token: bool = False
    # Server-only marker set exclusively by the MCP gateway admission path
    # (reload_admitted_user) for a keyless user-subject admitted via a gateway DCR session
    # bearer or bridge envelope. Not a DB column and never populated from caller-controlled key
    # metadata or JWT claims, so it cannot be forged to gain the team-inherited MCP grant union
    # or to escape the caller-Authorization egress scrub. exclude=True keeps it out of serialization.
    mcp_admitted_user_subject: bool = Field(default=False, exclude=True)
    requires_fresh_policy: bool = Field(default=False, exclude=True)
    mcp_explicit_grants_only: bool = Field(default=False, exclude=True)
    # team_id -> that team's mcp_rpm_limit map, for a keyless admitted subject that reaches MCP
    # servers through several teams at once and therefore has no single team_id for the limiter to
    # key off. Server-only and stripped from validated input for the same reason as the marker
    # above: a forged entry would let a caller pick which team's rpm bucket it is charged against.
    mcp_source_team_rpm_limits: dict[str, dict[str, int]] | None = Field(default=None, exclude=True)
    # The single MCP server_id a gateway session bearer was scoped to at authorize time (RFC 8707
    # resource), or None for an aggregate-scope session. A RESTRICTION intersected against the live
    # grant resolution, never a grant. Server-only, set exclusively by the MCP gateway admission
    # path via post-construction assignment and stripped from validated input like the markers
    # above; a forged value could at most narrow, but the stripping keeps the field's provenance
    # single-owner so its meaning stays trustworthy.
    mcp_session_resource_server_id: str | None = Field(default=None, exclude=True)
    mcp_toolset_id: str | None = Field(default=None, exclude=True)
    authenticated_by_custom_auth: bool = Field(default=False, exclude=True)
    via_virtual_key: bool = Field(
        default=False,
        exclude=True,
        description=(
            "Server-only marker set exclusively by the DB virtual-key and master-key auth paths via "
            "post-construction assignment. Stripped from validated input so custom auth handlers, JWT "
            "claims, or key metadata cannot forge it. Gates overwrite_user_with_key_hash stamping: only "
            "a credential the proxy itself validated as a key may be forwarded as the provider-facing "
            "user id."
        ),
    )
    invoked_agent_id: str | None = Field(default=None, exclude=True)
    invoked_agent_policy: AgentResponse | None = Field(default=None, exclude=True)
    agent_invocation_cost: float | None = Field(default=None, exclude=True)
    billing_agent_policy: AgentResponse | None = Field(default=None, exclude=True)
    _managed_delegation_verified: bool = PrivateAttr(default=False)
    managed_agent_policy: AgentResponse | None = Field(default=None, exclude=True)
    managed_agent_context: ManagedAgentContext | None = Field(default=None, exclude=True)
    agent_caller: AgentCaller | None = Field(
        default=None,
        exclude=True,
        description=(
            "Set per request from the x-litellm-user-id / x-litellm-team-id headers an agent echoes back on "
            "calls made with its own key. Every check treats it as a ceiling, so a forged value can only "
            "narrow the agent's access."
        ),
    )
    budget_reservation: dict[str, Any] | None = Field(default=None, exclude=True)
    team_budget_snapshot: TeamBudgetSnapshot | None = Field(default=None, exclude=True)
    user_budget_snapshot: UserBudgetSnapshot | None = Field(default=None, exclude=True)
    org_budget_snapshot: OrgBudgetSnapshot | None = Field(default=None, exclude=True)
    matched_model_access_groups: list[str] | None = Field(default=None, exclude=True)
    budget_throttle_pct: float | None = Field(default=None, exclude=True)
    user: Any | None = None  # Expanded user object when expand=user is used
    created_by_user: Any | None = None  # Expanded created_by user when expand=user is used
    end_user_object_permission: LiteLLM_ObjectPermissionTable | None = None
    # Team object_permission preloaded in auth (e.g. get_team_object) to avoid
    # per-request object_permission fetches in downstream checks (vector stores, etc.)
    team_object_permission: LiteLLM_ObjectPermissionTable | None = None
    # Decoded upstream IdP claims (groups, roles, etc.) propagated by JWT auth machinery
    # and forwarded into outbound tokens by guardrails such as MCPJWTSigner.
    jwt_claims: _UntypedDict | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def check_api_key(cls, values: object) -> object:
        # If values is already an instance (not a dict), return it as-is
        if not _is_str_keyed_dict(values):
            return values
        # mcp_admitted_user_subject is a server-only marker, set ONLY by the MCP gateway admission
        # path via post-construction assignment. Strip it from any validated input (constructor
        # kwargs, model_validate, a JWT/key claim splat) so it can never be forged from caller data.
        values.pop("mcp_admitted_user_subject", None)
        values.pop("requires_fresh_policy", None)
        values.pop("mcp_explicit_grants_only", None)
        values.pop("mcp_source_team_rpm_limits", None)
        values.pop("mcp_session_resource_server_id", None)
        values.pop("mcp_toolset_id", None)
        values.pop("via_virtual_key", None)
        values.pop("authenticated_by_custom_auth", None)
        values.pop("agent_caller", None)
        values.pop("managed_agent_context", None)
        values.pop("managed_agent_policy", None)
        values.pop("invoked_agent_id", None)
        values.pop("invoked_agent_policy", None)
        values.pop("agent_invocation_cost", None)
        values.pop("billing_agent_policy", None)
        api_key: Final = values.get("api_key")
        if api_key is not None:
            values.update({"token": cls._safe_hash_litellm_api_key(api_key)})  # pyright: ignore[reportArgumentType]  # non-str fails in the hasher
            if isinstance(api_key, str):
                values.update({"api_key": cls._safe_hash_litellm_api_key(api_key)})
        return values

    @classmethod
    def _safe_hash_litellm_api_key(cls, api_key: str) -> str:
        """
        Helper to ensure all logged keys are hashed
        Covers:
        1. Regular API keys from LiteLLM DB
        2. JWT tokens used for connecting to LiteLLM API
        """
        normalized = api_key
        if normalized[:7].lower() == "bearer ":
            normalized = normalized[7:]
        if normalized.startswith("sk-"):
            return hash_token(normalized)
        if is_jwt(token=normalized):
            return f"hashed-jwt-{hash_token(token=normalized)}"
        return normalized

    @classmethod
    def get_litellm_internal_health_check_user_api_key_auth(cls) -> "UserAPIKeyAuth":
        """
        Returns a `UserAPIKeyAuth` object for the litellm internal health check service account.

        This is used to track number of requests/spend for health check calls.
        """
        from litellm.constants import LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME

        return cls(
            api_key=LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
            team_id=LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
            key_alias=LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
            team_alias=LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
        )

    @classmethod
    def get_litellm_cli_user_api_key_auth(cls) -> "UserAPIKeyAuth":
        """
        Returns a `UserAPIKeyAuth` object for the litellm internal health check service account.

        This is used to track number of requests/spend for health check calls.
        """
        from litellm.constants import LITTELM_CLI_SERVICE_ACCOUNT_NAME

        return cls(
            api_key=LITTELM_CLI_SERVICE_ACCOUNT_NAME,
            team_id=LITTELM_CLI_SERVICE_ACCOUNT_NAME,
            key_alias=LITTELM_CLI_SERVICE_ACCOUNT_NAME,
            team_alias=LITTELM_CLI_SERVICE_ACCOUNT_NAME,
        )

    @classmethod
    def get_litellm_internal_jobs_user_api_key_auth(cls) -> "UserAPIKeyAuth":
        """
        Returns a `UserAPIKeyAuth` object for internal LiteLLM jobs like key rotation.

        This is used to track actions performed by automated system jobs.
        """
        from litellm.constants import LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME

        return cls(
            api_key=LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
            team_id="system",
            key_alias=LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
            team_alias="system",
            user_id="system",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )

    @property
    def is_team_service_account(self) -> bool:
        return (
            self.user_id is None
            and self.team_id is not None
            and bool(self.metadata)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped base field
            and self.metadata.get("service_account_id") is not None  # pyright: ignore[reportUnknownMemberType]  # untyped base field
        )


def user_api_key_has_admin_view(user_api_key_dict: UserAPIKeyAuth) -> bool:
    """Return True if the caller's role grants unscoped read access to all
    tenant resources (managed files, batches, vector stores, spend rows, etc).

    Lives on _types.py so leaf modules (e.g. litellm.llms.base_llm.managed_resources)
    can use it without pulling in litellm.proxy.utils via management_endpoints.
    """
    return user_api_key_dict.user_role in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    )
