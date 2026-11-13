use jiff::Timestamp;
use serde_json::{Map, Value};
use std::collections::BTreeSet;

use litellm_config::{ClaimPath, LitellmRole, RbacRole};

use super::{
    FinishedJwt, JwtDenial, JwtIdentity, JwtOutcome, JwtRequest, JwtRows, JwtSettings,
    claims::{self, UserInfo},
    mapping::{self, AfterAuthorize, BeforeAuthorize},
    rbac,
    team::{self, HeaderTeam, Team, TeamClaims, TeamSettings},
    verify::{self, Claims},
};
use crate::{
    AuthHost, Error, Rows,
    credential::key::Token,
    host::{AgentRow, MembershipRow, TeamRow, UserRow},
    routes::passthrough::PassthroughRequest,
};

/// `_JWTProvisioning`
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Provisioning {
    pub user_id_upsert: bool,
    pub team_id_upsert: bool,
}

impl From<&JwtSettings> for Provisioning {
    fn from(settings: &JwtSettings) -> Self {
        Self {
            user_id_upsert: settings.config.user_id_upsert,
            team_id_upsert: settings.config.team_id_upsert,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentMode {
    Autonomous,
    Delegated,
}

/// The `ManagedAgentContext` `resolve_managed_agent` found for the claims, with the context row
/// the core hands back to `record_agent_authentication` and in the identity.
#[derive(Clone, Debug, PartialEq)]
pub struct ManagedAgent<R> {
    pub mode: AgentMode,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub row: R,
}

/// The agent registered under a JWT's agent claim, by id and then by name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegisteredAgent {
    pub agent_id: String,
    pub identity_managed: bool,
    pub has_identity: bool,
    /// `has_legacy_identity(litellm_params)`
    pub has_legacy_identity: bool,
}

/// The arguments of `resolve_delegated_agent_team`.
#[derive(Clone, Copy, Debug)]
pub struct DelegatedTeam<'a> {
    pub user_id: Option<&'a str>,
    pub agent_id: &'a str,
    pub team_id: Option<&'a str>,
    pub explicit_team: bool,
    pub allowed_team_ids: Option<&'a BTreeSet<String>>,
}

/// What the core reads from a `LiteLLM_UserTable`, with the row it hands back to
/// `set_user_role` and `patch_team_membership` and in the identity.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct User<R> {
    pub user_id: String,
    pub user_email: Option<String>,
    pub user_role: Option<String>,
    pub teams: Vec<String>,
    /// `metadata.scim_active is False`
    pub deactivated: bool,
    pub row: R,
}

/// `get_user_object`: `Absent` is its `UserNotFoundError`.
#[derive(Clone, Debug, PartialEq)]
pub enum UserRead<R> {
    Found(User<R>),
    Absent,
}

impl<R> UserRead<R> {
    pub fn found(self) -> Option<User<R>> {
        match self {
            Self::Found(user) => Some(user),
            Self::Absent => None,
        }
    }
}

/// `GrantResolver.resolve_identity`: the user, and its membership of the team it was asked about.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentityRead<R: Rows> {
    pub user: UserRead<R::User>,
    pub membership: Option<R::Membership>,
}

/// `UserLookup`, the arguments `get_user_object` reads a user by.
#[derive(Clone, Copy, Debug)]
pub struct UserLookup<'a> {
    pub user_id: &'a str,
    pub user_email: Option<&'a str>,
    pub sso_user_id: Option<&'a str>,
    pub upsert: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Org<R> {
    pub organization_id: String,
    pub row: R,
}

/// `JWTIdentity`
#[derive(Clone, Debug, PartialEq)]
pub struct ClaimIdentity<R> {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub user: Option<R>,
}

pub(super) struct Authorized<R: Rows> {
    pub(super) identity: JwtIdentity<R>,
    pub(super) user: Option<User<R::User>>,
}

type HostUser<H> = User<UserRow<H>>;
type HostTeam<H> = Team<TeamRow<H>>;

/// `auth_jwt`, or `get_oidc_userinfo` for a token that is not a JWT when UserInfo is enabled.
async fn decode(
    host: &impl AuthHost,
    token: &Token,
    settings: &JwtSettings,
    now: Timestamp,
) -> Result<Claims, Error> {
    if settings.config.oidc_userinfo_enabled && !token.is_jwt() {
        let userinfo = verify::oidc_userinfo(host, token, settings).await?;
        return Ok(Claims(match userinfo {
            Value::Object(claims) => claims,
            _ => Map::new(),
        }));
    }
    verify::verify(host, token, settings, now).await
}

async fn custom_validate(
    host: &impl AuthHost,
    claims: &Claims,
    settings: &JwtSettings,
) -> Result<(), Error> {
    if !verify::operator_accepts(host, &claims.0, settings).await? {
        return Err(JwtDenial::CustomValidate.into());
    }
    Ok(())
}

/// `JWTAuthManager.authenticate_jwt`
pub(super) async fn authenticate_token(
    host: &impl AuthHost,
    token: &Token,
    settings: &JwtSettings,
    now: Timestamp,
) -> Result<Claims, Error> {
    let claims = decode(host, token, settings, now).await?;
    custom_validate(host, &claims, settings).await?;
    Ok(claims)
}

/// `JWTAuthManager.resolve_agent_id`
async fn resolve_agent_id(
    host: &impl AuthHost,
    claims: &Claims,
    settings: &JwtSettings,
) -> Result<Option<String>, Error> {
    let Some(claim) = claims::agent_claim(claims, settings) else {
        return Ok(None);
    };
    match host.registered_agent(&claim).await? {
        Some(agent)
            if !agent.identity_managed && !agent.has_identity && !agent.has_legacy_identity =>
        {
            Ok(Some(agent.agent_id))
        }
        _ => Err(JwtDenial::UnregisteredAgent {
            agent_id_jwt_field: settings
                .config
                .agent_id_jwt_field
                .as_ref()
                .map(ToString::to_string),
            claim,
        }
        .into()),
    }
}

/// What `GrantResolver.resolve_identity` resolved a claim's user to.
struct ResolvedUser<H: AuthHost> {
    user: Option<HostUser<H>>,
    /// The id the user rebinds the claim to.
    user_id: Option<String>,
    membership: Option<MembershipRow<H>>,
}

/// `GrantResolver.resolve_identity`
async fn resolve_user<H: AuthHost>(
    host: &H,
    user_id: Option<&str>,
    user_email: Option<&str>,
    upsert: bool,
    team_id: Option<&str>,
) -> Result<ResolvedUser<H>, Error> {
    let Some(user_id) = user_id.filter(|text| !text.is_empty()) else {
        return Ok(ResolvedUser {
            user: None,
            user_id: user_id.map(str::to_owned),
            membership: None,
        });
    };
    let lookup = UserLookup {
        user_id,
        user_email,
        sso_user_id: Some(user_id),
        upsert,
    };
    let read = host.resolve_user_identity(&lookup, team_id).await?;
    let Some(user) = read.user.found() else {
        return Err(JwtDenial::UserNotFound {
            user_id: user_id.to_owned(),
        }
        .into());
    };
    let effective = if user.user_id.is_empty() {
        user_id.to_owned()
    } else {
        user.user_id.clone()
    };
    Ok(ResolvedUser {
        user: Some(user),
        user_id: Some(effective),
        membership: read.membership,
    })
}

fn email_domain_denial(user_email: Option<&str>, settings: &JwtSettings) -> JwtDenial {
    JwtDenial::EmailDomainNotAllowed {
        user_email: user_email.map(str::to_owned),
        allowed_domain: settings.config.user_allowed_email_domain.clone(),
    }
}

/// `JWTAuthManager._resolve_claim_identity`, with the user it loaded.
pub(super) async fn claim_identity<H: AuthHost>(
    host: &H,
    claims: &Claims,
    settings: &JwtSettings,
) -> Result<(ClaimIdentity<UserRow<H>>, Option<HostUser<H>>), Error> {
    let info = claims::user_info(claims, settings);
    let user_id = match claims::rbac_role(claims, settings)? {
        Some(RbacRole::InternalUser) => claims::object_id(claims, settings)
            .filter(|text| !text.is_empty())
            .or(info.user_id),
        _ => info.user_id,
    };
    let agent_id = resolve_agent_id(host, claims, settings).await?;
    let is_admin = claims::is_admin(&claims::scopes(claims)?, settings);
    if info.valid_user_email == Some(false) {
        return Err(email_domain_denial(info.user_email.as_deref(), settings).into());
    }
    match resolve_user(
        host,
        user_id.as_deref(),
        info.user_email.as_deref(),
        false,
        None,
    )
    .await
    {
        Ok(resolved) => Ok((
            ClaimIdentity {
                user_id: if is_admin { user_id } else { resolved.user_id },
                agent_id,
                user: resolved.user.as_ref().map(|user| user.row.clone()),
            },
            resolved.user,
        )),
        Err(Error::Jwt(JwtDenial::UserNotFound { .. })) if is_admin => Ok((
            ClaimIdentity {
                user_id,
                agent_id,
                user: None,
            },
            None,
        )),
        Err(error) => Err(error),
    }
}

/// The membership changes `sync_user_role_and_teams` makes from the token's teams.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TeamSync {
    pub add: Vec<String>,
    pub remove: Vec<String>,
    /// The user's teams afterwards.
    pub teams: Vec<String>,
}

fn unique(ids: impl IntoIterator<Item = String>) -> Vec<String> {
    ids.into_iter().fold(Vec::new(), |mut seen, id| {
        if !seen.contains(&id) {
            seen.push(id);
        }
        seen
    })
}

/// The team half of `sync_user_role_and_teams`. Under `fallback_to_db_teams` a token without
/// team claims keeps the user's DB teams. `remove` follows the user's team order.
pub fn team_sync(claims: &Claims, user_teams: &[String], settings: &JwtSettings) -> TeamSync {
    let fallback = settings.config.fallback_to_db_teams;
    let claimed = unique(match fallback {
        true => claims::all_team_ids(claims, settings),
        false => claims::team_ids(claims, settings),
    });
    let existing = unique(user_teams.iter().cloned());
    let add: Vec<String> = claimed
        .iter()
        .filter(|id| !existing.contains(id))
        .cloned()
        .collect();
    let remove: Vec<String> = match fallback && claimed.is_empty() {
        true => Vec::new(),
        false => existing
            .iter()
            .filter(|id| !claimed.contains(id))
            .cloned()
            .collect(),
    };
    let teams = existing
        .iter()
        .filter(|id| !remove.contains(id))
        .chain(&add)
        .cloned()
        .collect();
    TeamSync { add, remove, teams }
}

/// `JWTAuthManager.sync_user_role_and_teams`
async fn sync_user<H: AuthHost>(
    host: &H,
    claims: &Claims,
    user: HostUser<H>,
    settings: &JwtSettings,
) -> Result<HostUser<H>, Error> {
    let new_role = claims::litellm_role(claims, settings)
        .filter(|role| user.user_role.as_deref() != Some(<&str>::from(*role)));
    let user = match new_role {
        Some(role) => User {
            row: host.set_user_role(&user.row, role).await?,
            user_role: Some(<&str>::from(role).to_owned()),
            ..user
        },
        None => user,
    };
    let sync = team_sync(claims, &user.teams, settings);
    if sync.add.is_empty() && sync.remove.is_empty() {
        return Ok(user);
    }
    Ok(User {
        row: host
            .patch_team_membership(&user.row, &sync.add, &sync.remove)
            .await?,
        teams: sync.teams,
        ..user
    })
}

/// What `authorize_jwt` requires of a managed agent's token before it reads any other claim.
async fn admit_managed_agent<H: AuthHost>(
    host: &H,
    claims: &Claims,
    agent: &ManagedAgent<AgentRow<H>>,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
) -> Result<(), Error> {
    if !verify::managed_issuer_is_trusted(claims.0.get("iss").unwrap_or(&Value::Null), settings) {
        return Err(JwtDenial::ManagedIssuerUntrusted.into());
    }
    if !settings
        .route_groups
        .managed_agent_route_allowed(request.route, request.method)
    {
        return Err(JwtDenial::ManagedRouteNotAllowed.into());
    }
    if let Some(failure) = host.record_agent_authentication(&agent.row).await? {
        return Err(JwtDenial::AgentIdentity(failure).into());
    }
    Ok(())
}

/// An autonomous managed agent acts as itself, with no user, team or organization.
fn autonomous_identity<R: Rows>(claims: Value, agent: ManagedAgent<R::Agent>) -> JwtIdentity<R> {
    JwtIdentity {
        claims,
        is_proxy_admin: false,
        user_id: None,
        user_email: None,
        team_id: None,
        org_id: None,
        end_user_id: None,
        agent_id: Some(agent.agent_id.clone()),
        managed_agent: Some(agent),
        rows: JwtRows::default(),
    }
}

/// The user row a proxy admin's identity carries.
async fn admin_user<H: AuthHost>(
    host: &H,
    claims: &Claims,
    settings: &JwtSettings,
    provisioning: Option<Provisioning>,
    user_id: Option<&str>,
    user_email: Option<&str>,
) -> Result<Option<HostUser<H>>, Error> {
    match (provisioning, user_id) {
        (None, _) => Ok(claim_identity(host, claims, settings).await?.1),
        (Some(_), _) if !settings.has_db => Ok(None),
        (Some(_), None) => Ok(None),
        (Some(_), Some(id)) => {
            let lookup = UserLookup {
                user_id: id,
                user_email,
                sso_user_id: Some(id),
                upsert: false,
            };
            Ok(host.user(&lookup).await?.found())
        }
    }
}

/// The team `authorize_jwt` takes from the team header, the token's claims or an RBAC team token,
/// before a managed agent's delegation or a fallback to the user's own teams.
struct ClaimedTeam<H: AuthHost> {
    team_id: Option<String>,
    /// Its row, when resolving it read one.
    team: Option<HostTeam<H>>,
    header_team: Option<HeaderTeam>,
    /// The token names no team, so one of the user's DB teams is picked later.
    db_fallback: bool,
    /// The teams a managed agent's delegation may act for.
    scope: Option<BTreeSet<String>>,
}

async fn claimed_team<H: AuthHost>(
    host: &H,
    claims: &Claims,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
    rbac_team_id: Option<String>,
    team_id_upsert: bool,
    passthrough: &PassthroughRequest<'_>,
) -> Result<ClaimedTeam<H>, Error> {
    let team_settings = TeamSettings::from(&settings.config);
    let claim_team_ids = claims::all_team_ids(claims, settings);
    let team_alias = claims::team_alias(claims, settings).filter(|text| !text.is_empty());
    let specific_team_id = claims::team_id(claims, settings);
    let db_fallback = settings.config.fallback_to_db_teams
        && claim_team_ids.is_empty()
        && team_alias.is_none()
        && rbac_team_id.is_none();
    let team_ids: BTreeSet<String> = claims::team_ids(claims, settings)
        .into_iter()
        .chain(
            specific_team_id
                .clone()
                .filter(|text| !text.is_empty())
                .filter(|_| !db_fallback),
        )
        .collect();
    let header_db_fallback = settings.config.fallback_to_db_teams && rbac_team_id.is_none();
    let header_team =
        team::header_team(host, request.team_header, &team_ids, header_db_fallback).await?;

    let (team_id, team) = match &header_team {
        Some(header_team) => {
            let loaded = team::load_header_team(host, header_team, team_id_upsert).await?;
            (Some(header_team.team_id.clone()), Some(loaded))
        }
        None if rbac_team_id.is_none() && !db_fallback => {
            let team_claims = TeamClaims {
                team_id: specific_team_id.as_deref(),
                has_team_id_claim: !claim_team_ids.is_empty(),
                team_alias: team_alias.as_deref(),
                team_id_jwt_field: settings
                    .config
                    .team_id_jwt_field
                    .as_ref()
                    .map(ClaimPath::as_str),
                team_alias_jwt_field: settings
                    .config
                    .team_alias_jwt_field
                    .as_ref()
                    .map(ClaimPath::as_str),
            };
            let found =
                team::specific_team(host, team_claims, team_id_upsert, team_settings).await?;
            (found.as_ref().map(|team| team.team_id.clone()), found)
        }
        None => (rbac_team_id, None),
    };
    let (team_id, team) = match team_id {
        None => {
            let route_listed = rbac::team_route_listed(request.route, settings);
            let found = team::select_team(
                host,
                &team_ids,
                request.model,
                passthrough,
                route_listed,
                team_settings,
            )
            .await?;
            (found.as_ref().map(|team| team.team_id.clone()), found)
        }
        Some(team_id) => (Some(team_id), team),
    };
    let team = match team_id.as_deref() {
        Some(id) if team.is_none() && request.auth_enforced_pass_through => {
            Some(host.team(id, team_id_upsert).await?.or_not_found(id)?)
        }
        _ => team,
    };
    let scope = match (&team_id, &team_alias) {
        _ if !claim_team_ids.is_empty() => Some(claim_team_ids.iter().cloned().collect()),
        (Some(team_id), Some(_)) => Some(BTreeSet::from([team_id.clone()])),
        _ => None,
    };
    Ok(ClaimedTeam {
        team_id,
        team,
        header_team,
        db_fallback,
        scope,
    })
}

/// `resolve_delegated_agent_team` for a managed agent. Any other token keeps the team it claimed.
async fn delegated_team<H: AuthHost>(
    host: &H,
    agent: Option<&ManagedAgent<AgentRow<H>>>,
    claimed: &ClaimedTeam<H>,
    settings: &JwtSettings,
) -> Result<Option<String>, Error> {
    let Some(agent) = agent else {
        return Ok(claimed.team_id.clone());
    };
    let delegated = DelegatedTeam {
        user_id: agent.user_id.as_deref(),
        agent_id: &agent.agent_id,
        team_id: claimed.team_id.as_deref(),
        explicit_team: claimed.header_team.is_some(),
        allowed_team_ids: claimed
            .scope
            .as_ref()
            .filter(|_| !settings.config.fallback_to_db_teams),
    };
    host.delegated_agent_team(&delegated).await
}

/// `JWTAuthManager.authorize_jwt` after `authenticate_jwt` and `resolve_managed_agent`.
pub(super) async fn authorize<H: AuthHost>(
    host: &H,
    claims: &Claims,
    managed: Option<ManagedAgent<AgentRow<H>>>,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
    provisioning: Option<Provisioning>,
) -> Result<Authorized<H::Rows>, Error> {
    let route = request.route;
    let claims_value = Value::Object(claims.0.clone());
    if let Some(agent) = &managed {
        admit_managed_agent(host, claims, agent, request, settings).await?;
    }
    let managed = match managed {
        Some(agent) if agent.mode == AgentMode::Autonomous => {
            let identity = autonomous_identity::<H::Rows>(claims_value, agent);
            return Ok(Authorized {
                identity,
                user: None,
            });
        }
        managed => managed,
    };
    let provisioned = provisioning.filter(|_| managed.is_none());
    let team_id_upsert = provisioning.is_some_and(|provisioning| provisioning.team_id_upsert);
    let model = request.model;
    let team_settings = TeamSettings::from(&settings.config);

    let rbac_role = match managed {
        None => claims::rbac_role(claims, settings)?,
        Some(_) => None,
    };
    rbac::check_rbac_role(rbac_role, model, route, settings)?;
    let scopes = claims::scopes(claims)?;
    if let Some(mappings) = settings.enforced_scope_mappings() {
        rbac::check_scope_based_access(mappings, &scopes, model)?;
    }

    let UserInfo {
        user_id,
        user_email,
        valid_user_email,
    } = match &managed {
        Some(agent) => UserInfo {
            user_id: agent.user_id.clone(),
            ..UserInfo::default()
        },
        None => claims::user_info(claims, settings),
    };
    let org_id = claims::org_id(claims, settings);
    let end_user_id = claims::end_user_id(claims, settings);
    let object_id = claims::object_id(claims, settings).filter(|text| !text.is_empty());
    let (rbac_team_id, user_id) = match (rbac_role, object_id) {
        (Some(RbacRole::Team), Some(object_id)) => (Some(object_id), user_id),
        (Some(RbacRole::InternalUser), Some(object_id)) => (None, Some(object_id)),
        _ => (None, user_id),
    };
    let agent_id = match &managed {
        Some(agent) => Some(agent.agent_id.clone()),
        None => resolve_agent_id(host, claims, settings).await?,
    };

    if managed.is_none() && claims::is_admin(&scopes, settings) {
        rbac::check_admin_route(route, settings)?;
        let team = team::admin_header_team(
            host,
            request.team_header,
            request.is_llm_api_route,
            team_id_upsert,
        )
        .await;
        let user = admin_user(
            host,
            claims,
            settings,
            provisioning,
            user_id.as_deref(),
            user_email.as_deref(),
        )
        .await?;
        let identity = JwtIdentity {
            claims: claims_value,
            is_proxy_admin: true,
            user_id,
            user_email,
            team_id: team.as_ref().map(|team| team.team_id.clone()),
            org_id,
            end_user_id: None,
            agent_id,
            managed_agent: None,
            rows: JwtRows {
                user: user.as_ref().map(|user| user.row.clone()),
                team: team.map(|team| team.row),
                ..JwtRows::default()
            },
        };
        return Ok(Authorized { identity, user });
    }

    let passthrough = PassthroughRequest {
        route,
        auth_enforced: request.auth_enforced_pass_through,
        forwarded_routes: request.forwarded_routes,
        jwt_team_allowed_routes: Some(settings.team_routes()),
    };
    let claimed = claimed_team(
        host,
        claims,
        request,
        settings,
        rbac_team_id,
        team_id_upsert,
        &passthrough,
    )
    .await?;

    let granting_team = delegated_team(host, managed.as_ref(), &claimed, settings).await?;
    let team_route_allowed =
        rbac::is_team_route_allowed(route, request.auth_enforced_pass_through, settings);
    if granting_team.is_some() && granting_team != claimed.team_id && !team_route_allowed.0 {
        return Err(JwtDenial::GrantingTeamRouteNotAllowed.into());
    }
    let selected_team_id = granting_team.or_else(|| claimed.team_id.clone());
    let selected_team = match selected_team_id.as_deref() {
        Some(id) if selected_team_id != claimed.team_id => {
            Some(host.granting_team(id).await?.or_not_found(id)?)
        }
        _ => claimed.team,
    };
    team::passthrough_gate(
        selected_team_id.as_deref(),
        selected_team.as_ref(),
        &passthrough,
    )?;

    let org = match (
        org_id.as_deref().filter(|text| !text.is_empty()),
        claims::org_alias(claims, settings).filter(|text| !text.is_empty()),
    ) {
        (Some(org_id), _) => host.org(org_id).await?,
        (None, Some(alias)) => host.org_by_alias(&alias).await?,
        (None, None) => None,
    };
    if valid_user_email == Some(false) {
        return Err(email_domain_denial(user_email.as_deref(), settings).into());
    }
    let upsert = provisioned.is_some_and(|provisioning| provisioning.user_id_upsert);
    let ResolvedUser {
        user,
        user_id,
        membership: selected_membership,
    } = resolve_user(
        host,
        user_id.as_deref(),
        user_email.as_deref(),
        upsert,
        selected_team_id.as_deref(),
    )
    .await?;
    let end_user = match end_user_id.as_deref().filter(|text| !text.is_empty()) {
        Some(end_user_id) => Some(host.jwt_end_user(end_user_id).await?),
        None => None,
    };
    let (org_id, org) = match org {
        Some(org) => (Some(org.organization_id), Some(org.row)),
        None => (org_id, None),
    };

    let user = match user {
        Some(user)
            if provisioned.is_some()
                && settings.config.sync_user_role_and_teams
                && settings.has_db =>
        {
            Some(sync_user(host, claims, user, settings).await?)
        }
        user => user,
    };
    let user_teams = user
        .as_ref()
        .map(|user| user.teams.as_slice())
        .unwrap_or_default();

    let (team_id, team, membership) = match selected_team_id {
        None if claimed.db_fallback => {
            let found = team::db_team_fallback(
                host,
                user_teams,
                user_id.as_deref(),
                model,
                &passthrough,
                team_route_allowed,
                team_id_upsert,
                team_settings,
            )
            .await?;
            if let Some(found) = &found {
                team::passthrough_gate(Some(&found.team.team_id), Some(&found.team), &passthrough)?;
            }
            joined(found)
        }
        None => joined(
            team::single_team_fallback(host, user_teams, user_id.as_deref(), team_id_upsert)
                .await?,
        ),
        Some(selected) => {
            if let Some(header_team) = claimed
                .header_team
                .as_ref()
                .filter(|header_team| header_team.provisional)
                && header_team.team_id == selected
            {
                team::check_provisional_header_team(
                    header_team,
                    user_teams,
                    team_route_allowed,
                    route,
                )?;
            }
            (Some(selected), selected_team, selected_membership)
        }
    };

    if provisioned.is_some()
        && let Some(user) = &user
        && let Some(team) = &team
        && !team.members.contains(&user.user_id)
    {
        host.add_team_member(&user.user_id, &team.team_id).await?;
    }

    if settings.enforce_rbac
        && user_id.as_deref().is_none_or(str::is_empty)
        && team_id.as_deref().is_none_or(str::is_empty)
    {
        return Err(JwtDenial::NoObjectId.into());
    }

    let is_proxy_admin = managed.is_none()
        && user.as_ref().is_some_and(|user| {
            user.user_role.as_deref() == Some(<&str>::from(LitellmRole::ProxyAdmin))
        });
    let identity = JwtIdentity {
        claims: claims_value,
        is_proxy_admin,
        user_id,
        user_email: user
            .as_ref()
            .and_then(|user| user.user_email.clone().filter(|text| !text.is_empty()))
            .or(user_email),
        team_id,
        org_id,
        end_user_id,
        agent_id,
        managed_agent: managed,
        rows: JwtRows {
            user: user.as_ref().map(|user| user.row.clone()),
            team: team.map(|team| team.row),
            membership: membership.filter(|_| !is_proxy_admin),
            org,
            end_user,
        },
    };
    Ok(Authorized { identity, user })
}

/// A team's id, the team and the membership loaded with it.
type JoinedTeam<R> = (
    Option<String>,
    Option<Team<<R as Rows>::Team>>,
    Option<<R as Rows>::Membership>,
);

/// The team a fallback joined and the membership it loaded with it.
fn joined<R: Rows>(found: Option<team::Joined<R>>) -> JoinedTeam<R> {
    match found {
        Some(joined) => (
            Some(joined.team.team_id.clone()),
            Some(joined.team),
            joined.membership,
        ),
        None => (None, None, None),
    }
}

/// The builder's JWT block. Claims verified for the mapping are reused by `auth_builder`.
pub(super) async fn build<H: AuthHost>(
    host: &H,
    token: &Token,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
) -> Result<JwtOutcome<H::Rows>, Error> {
    let claims = decode(host, token, settings, request.now).await?;
    let pending = match mapping::before_authorize(host, &claims, settings).await? {
        BeforeAuthorize::Mapped(outcome) => return Ok(*outcome),
        BeforeAuthorize::Pending(pending) => Some(pending),
        BeforeAuthorize::Unmapped => None,
    };

    custom_validate(host, &claims, settings).await?;
    let managed = host.managed_agent(&Value::Object(claims.0.clone())).await?;
    let Authorized { identity, user } = authorize(
        host,
        &claims,
        managed,
        request,
        settings,
        Some(settings.into()),
    )
    .await?;
    if user.is_some_and(|user| user.deactivated) {
        return Err(JwtDenial::UserDeactivated {
            user_id: identity.user_id,
        }
        .into());
    }
    let identity = match &pending {
        Some(pending) => match mapping::after_authorize(host, pending, identity, settings).await? {
            AfterAuthorize::Mapped(outcome) => return Ok(outcome),
            AfterAuthorize::Unmapped(identity) => identity,
        },
        None => identity,
    };
    if identity.is_proxy_admin {
        return Ok(JwtOutcome::Admin(identity));
    }
    Ok(JwtOutcome::Finished(
        host.finish_jwt(FinishedJwt::Identity(&identity)).await?,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use litellm_config::{
        JwtAuthSettings, JwtLitellmRoleMap, RoleMapping, ScopeMapping,
        UnregisteredJwtClientBehavior,
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::{
        BudgetExceeded, ErrorType, PublicDenial,
        credential::jwt::AgentIdentityFailure,
        credential::jwt::{
            AliasRead, RbacDenial, Team, TeamDenial, TeamRead, VerifyDenial,
            mapping::{KeyOrigin, MappedKey, NO_MAPPING_SENTINEL, PROXY_ADMIN_SENTINEL},
        },
        credential::key::{KeyHash, KeyRecord, Keyed},
        routes::passthrough::{ForwardedRoutes, PassthroughGrants},
        test_support::route_groups,
        testing::{Labels, ScriptedHost},
    };

    static POST: http::Method = http::Method::POST;
    const CHAT: &str = "/chat/completions";
    const TOKEN: &str = "opaque";

    fn config() -> JwtAuthSettings {
        JwtAuthSettings {
            oidc_userinfo_enabled: true,
            oidc_userinfo_endpoint: Some("https://idp/userinfo".to_owned()),
            user_id_jwt_field: Some("sub".into()),
            user_email_jwt_field: Some("email".into()),
            ..JwtAuthSettings::default()
        }
    }

    fn settings(config: JwtAuthSettings) -> JwtSettings {
        JwtSettings {
            config,
            has_db: true,
            has_master_key: true,
            issuer: Some("https://idp".to_owned()),
            audience: Some("litellm".to_owned()),
            route_groups: route_groups(),
            ..JwtSettings::default()
        }
    }

    fn request(route: &'static str) -> JwtRequest<'static> {
        JwtRequest {
            route,
            method: Some(&POST),
            ..JwtRequest::default()
        }
    }

    fn host(claims: Value) -> ScriptedHost {
        ScriptedHost {
            userinfo: Some(claims),
            ..ScriptedHost::default()
        }
    }

    fn team_with(team_id: &str, passthrough: PassthroughGrants) -> Result<TeamRead<String>, Error> {
        Ok(TeamRead::Found(Team {
            team_id: team_id.to_owned(),
            models: Some(Vec::new()),
            passthrough,
            members: Vec::new(),
            row: team_id.to_owned(),
        }))
    }

    fn team(team_id: &str) -> Result<TeamRead<String>, Error> {
        team_with(team_id, PassthroughGrants::default())
    }

    fn team_of(team_id: &str, members: &[&str]) -> Result<TeamRead<String>, Error> {
        Ok(TeamRead::Found(Team {
            team_id: team_id.to_owned(),
            models: Some(Vec::new()),
            members: members.iter().map(|member| (*member).to_owned()).collect(),
            row: team_id.to_owned(),
            ..Team::default()
        }))
    }

    fn user(user_id: &str, role: &str, teams: &[&str]) -> User<String> {
        User {
            user_id: user_id.to_owned(),
            user_email: None,
            user_role: Some(role.to_owned()),
            teams: teams.iter().map(|team| (*team).to_owned()).collect(),
            deactivated: false,
            row: user_id.to_owned(),
        }
    }

    /// The calls after the OIDC UserInfo read every token here goes through.
    fn calls(host: &ScriptedHost) -> Vec<String> {
        host.calls()
            .into_iter()
            .filter(|call| {
                !call.starts_with("get oidc_userinfo_")
                    && !call.starts_with("set oidc_userinfo_")
                    && !call.starts_with("userinfo ")
            })
            .collect()
    }

    async fn authorize_with(
        host: &ScriptedHost,
        request: JwtRequest<'_>,
        settings: &JwtSettings,
        provisioning: Option<Provisioning>,
    ) -> Result<JwtIdentity<Labels>, Error> {
        crate::credential::jwt::authorize_jwt(
            host,
            &Token::new(TOKEN),
            &request,
            settings,
            provisioning,
        )
        .await
    }

    const PROVISIONED: Option<Provisioning> = Some(Provisioning {
        user_id_upsert: false,
        team_id_upsert: false,
    });

    #[rstest]
    #[case::management_route_ignores_header("/key/info", false, PROVISIONED, None, &["managed_agent", "user u1 -"])]
    #[case::llm_route_attaches_header_team(
        CHAT,
        true,
        PROVISIONED,
        Some("t1"),
        &["managed_agent", "team t1", "user u1 -"]
    )]
    #[case::mcp_flow_resolves_the_claim_identity(
        "/key/info",
        false,
        None,
        None,
        &["managed_agent", "resolve_user u1 team=-"]
    )]
    #[tokio::test]
    async fn admin_scope(
        #[case] route: &'static str,
        #[case] is_llm_api_route: bool,
        #[case] provisioning: Option<Provisioning>,
        #[case] team_id: Option<&str>,
        #[case] expected_calls: &[&str],
    ) {
        let host = ScriptedHost {
            teams: HashMap::from([("t1", team("t1"))]),
            users: HashMap::from([("u1", user("u1", "internal_user", &[]))]),
            ..host(json!({"sub": "u1", "scope": "litellm_proxy_admin"}))
        };
        let settings = settings(JwtAuthSettings {
            admin_allowed_routes: Box::new(["info_routes".to_owned(), "openai_routes".to_owned()]),
            ..config()
        });
        let request = JwtRequest {
            team_header: Some("t1"),
            is_llm_api_route,
            ..request(route)
        };
        let identity = authorize_with(&host, request, &settings, provisioning)
            .await
            .unwrap();
        assert!(identity.is_proxy_admin);
        assert_eq!(identity.user_id.as_deref(), Some("u1"));
        assert_eq!(identity.team_id.as_deref(), team_id);
        assert_eq!(calls(&host), expected_calls);
    }

    #[tokio::test]
    async fn admin_scope_outside_admin_routes_is_denied() {
        let host = host(json!({"sub": "u1", "scope": "litellm_proxy_admin"}));
        let denial = authorize_with(&host, request(CHAT), &settings(config()), PROVISIONED)
            .await
            .unwrap_err();
        assert!(matches!(
            denial,
            Error::Jwt(JwtDenial::Rbac(RbacDenial::AdminRoute { .. }))
        ));
        assert_eq!(calls(&host), ["managed_agent"]);
    }

    #[rstest]
    #[case::adds_the_user_to_the_claimed_team(
        PROVISIONED,
        &[],
        &[
            "managed_agent",
            "team t1",
            "resolve_user u1 team=t1",
            "add_member u1 t1",
        ]
    )]
    #[case::upserts_the_team_and_skips_existing_members(
        Some(Provisioning { user_id_upsert: true, team_id_upsert: true }),
        &["u1"],
        &["managed_agent", "team t1 upsert", "resolve_user u1 team=t1 upsert"]
    )]
    #[case::mcp_flow_never_provisions(
        None,
        &[],
        &["managed_agent", "team t1", "resolve_user u1 team=t1"]
    )]
    #[tokio::test]
    async fn team_claim(
        #[case] provisioning: Option<Provisioning>,
        #[case] members: &[&str],
        #[case] expected_calls: &[&str],
    ) {
        let host = ScriptedHost {
            teams: HashMap::from([("t1", team_of("t1", members))]),
            users: HashMap::from([("u1", user("u1", "internal_user", &["t1"]))]),
            ..host(json!({"sub": "u1", "team": "t1"}))
        };
        let settings = settings(JwtAuthSettings {
            team_id_jwt_field: Some("team".into()),
            ..config()
        });
        let identity = authorize_with(&host, request(CHAT), &settings, provisioning)
            .await
            .unwrap();
        assert_eq!(
            (identity.team_id.as_deref(), identity.is_proxy_admin),
            (Some("t1"), false)
        );
        assert_eq!(
            identity.rows,
            JwtRows {
                user: Some("u1".to_owned()),
                team: Some("t1".to_owned()),
                membership: Some("membership u1 t1".to_owned()),
                ..JwtRows::default()
            }
        );
        assert_eq!(calls(&host), expected_calls);
    }

    #[rstest]
    #[case::claimed_team(
        "t2",
        Ok(Some("t2")),
        &["managed_agent", "team t2", "resolve_user u1 team=t2"]
    )]
    #[case::team_outside_the_token(
        "t9",
        Err(JwtDenial::Team(TeamDenial::HeaderTeamNotAllowed {
            header_value: "t9".to_owned(),
            allowed_team_ids: ["t1".to_owned(), "t2".to_owned()].into(),
        }).into()),
        &["managed_agent", "alias t9"]
    )]
    #[tokio::test]
    async fn header_team(
        #[case] header: &'static str,
        #[case] expected: Result<Option<&str>, Error>,
        #[case] expected_calls: &[&str],
    ) {
        let host = ScriptedHost {
            teams: HashMap::from([("t1", team("t1")), ("t2", team("t2"))]),
            users: HashMap::from([("u1", user("u1", "internal_user", &[]))]),
            ..host(json!({"sub": "u1", "groups": ["t1", "t2"]}))
        };
        let settings = settings(JwtAuthSettings {
            team_ids_jwt_field: Some("groups".into()),
            ..config()
        });
        let request = JwtRequest {
            team_header: Some(header),
            ..request(CHAT)
        };
        let identity = authorize_with(&host, request, &settings, None).await;
        assert_eq!(
            identity
                .as_ref()
                .map(|identity| identity.team_id.as_deref()),
            expected.as_ref().map(|team| *team)
        );
        assert_eq!(calls(&host), expected_calls);
    }

    #[rstest]
    #[case::member_team_is_picked(
        &["t1"],
        Ok(Some("t1")),
        &[
            "managed_agent",
            "resolve_user u1 team=-",
            "team t1",
            "membership u1 t1",
            "add_member u1 t1",
        ]
    )]
    #[case::no_membership_under_enforcement(
        &[],
        Err(JwtDenial::Team(TeamDenial::NotTeamMember).into()),
        &["managed_agent", "resolve_user u1 team=-"]
    )]
    #[tokio::test]
    async fn db_team_fallback_runs_after_the_user_loads(
        #[case] user_teams: &[&str],
        #[case] expected: Result<Option<&str>, Error>,
        #[case] expected_calls: &[&str],
    ) {
        let host = ScriptedHost {
            teams: HashMap::from([("t1", team("t1"))]),
            users: HashMap::from([("u1", user("u1", "internal_user", user_teams))]),
            ..host(json!({"sub": "u1"}))
        };
        let settings = settings(JwtAuthSettings {
            fallback_to_db_teams: true,
            enforce_team_based_model_access: true,
            ..config()
        });
        let identity = authorize_with(&host, request(CHAT), &settings, PROVISIONED).await;
        assert_eq!(
            identity
                .as_ref()
                .map(|identity| identity.team_id.as_deref()),
            expected.as_ref().map(|team| *team)
        );
        assert_eq!(calls(&host), expected_calls);
    }

    #[tokio::test]
    async fn email_domain_is_checked_after_the_org_and_before_the_user() {
        let host = host(json!({"email": "a@evil.com", "org": "o1"}));
        let settings = settings(JwtAuthSettings {
            user_allowed_email_domain: Some("corp.com".to_owned()),
            org_id_jwt_field: Some("org".into()),
            ..config()
        });
        let denial = authorize_with(&host, request(CHAT), &settings, PROVISIONED)
            .await
            .unwrap_err();
        assert_eq!(
            denial.public(),
            Some(PublicDenial::Proxy {
                status: 403,
                message:
                    "Email domain not allowed. User email: a@evil.com. Allowed domain: corp.com"
                        .to_owned(),
                error_type: ErrorType::Auth,
                param: Some("user_email".to_owned()),
            })
        );
        assert_eq!(calls(&host), ["managed_agent", "org o1"]);
    }

    #[rstest]
    #[case::plural_claims_replace_db_teams(false, json!({"groups": ["t2", "t2"]}), &["t1", "t2", "t1"], &[], &["t1"], &["t2"])]
    #[case::adds_and_removes(false, json!({"groups": ["t3"]}), &["t1"], &["t3"], &["t1"], &["t3"])]
    #[case::no_claims_without_fallback_drops_db_teams(false, json!({}), &["t1"], &[], &["t1"], &[])]
    #[case::no_claims_under_fallback_keeps_db_teams(true, json!({}), &["t1"], &[], &[], &["t1"])]
    #[case::fallback_reads_the_singular_claim(true, json!({"team": "t2"}), &["t1"], &["t2"], &["t1"], &["t2"])]
    fn team_sync_diff(
        #[case] fallback_to_db_teams: bool,
        #[case] claims: Value,
        #[case] user_teams: &[&str],
        #[case] add: &[&str],
        #[case] remove: &[&str],
        #[case] teams: &[&str],
    ) {
        let settings = settings(JwtAuthSettings {
            team_ids_jwt_field: Some("groups".into()),
            team_id_jwt_field: Some("team".into()),
            fallback_to_db_teams,
            ..config()
        });
        let owned = |ids: &[&str]| ids.iter().map(|id| (*id).to_owned()).collect::<Vec<_>>();
        let sync = team_sync(
            &Claims(claims.as_object().unwrap().clone()),
            &owned(user_teams),
            &settings,
        );
        assert_eq!(
            sync,
            TeamSync {
                add: owned(add),
                remove: owned(remove),
                teams: owned(teams),
            }
        );
    }

    #[tokio::test]
    async fn sync_applies_the_role_and_teams_before_the_team_fallbacks_read_them() {
        let host = ScriptedHost {
            teams: HashMap::from([("t2", team("t2"))]),
            users: HashMap::from([("u1", user("u1", "internal_user", &["t1"]))]),
            ..host(json!({"sub": "u1", "roles": ["admins"], "groups": ["t2"]}))
        };
        let settings = settings(JwtAuthSettings {
            sync_user_role_and_teams: true,
            roles_jwt_field: Some("roles".into()),
            jwt_litellm_role_map: Some(Box::new([JwtLitellmRoleMap {
                jwt_role: "admin*".to_owned(),
                litellm_role: LitellmRole::ProxyAdmin,
            }])),
            team_ids_jwt_field: Some("groups".into()),
            ..config()
        });
        let identity = authorize_with(&host, request(CHAT), &settings, PROVISIONED)
            .await
            .unwrap();
        assert!(identity.is_proxy_admin);
        assert_eq!(
            calls(&host),
            [
                "managed_agent",
                "team t2",
                "resolve_user u1 team=t2",
                "set_role u1 proxy_admin",
                "patch u1+role +t2 -t1",
                "add_member u1 t2",
            ]
        );
        assert_eq!(identity.rows.user.as_deref(), Some("u1+role+teams"));
        assert_eq!(
            identity.rows.membership, None,
            "a proxy admin's identity carries no membership"
        );
    }

    #[rstest]
    #[case::no_user_or_team(json!({}), Err(JwtDenial::NoObjectId.into()), &["managed_agent"])]
    #[case::user_is_enough(json!({"sub": "u1"}), Ok(Some("u1")), &["managed_agent", "resolve_user u1 team=-"])]
    #[tokio::test]
    async fn enforce_rbac_needs_an_object_id(
        #[case] claims: Value,
        #[case] expected: Result<Option<&str>, Error>,
        #[case] expected_calls: &[&str],
    ) {
        let host = ScriptedHost {
            users: HashMap::from([("u1", user("u1", "internal_user", &[]))]),
            ..host(claims)
        };
        let settings = JwtSettings {
            enforce_rbac: true,
            ..settings(config())
        };
        let identity = authorize_with(&host, request(CHAT), &settings, PROVISIONED).await;
        assert_eq!(
            identity
                .as_ref()
                .map(|identity| identity.user_id.as_deref()),
            expected.as_ref().map(|user| *user)
        );
        assert_eq!(calls(&host), expected_calls);
    }

    fn agent(mode: AgentMode) -> ManagedAgent<String> {
        ManagedAgent {
            mode,
            agent_id: "a1".to_owned(),
            user_id: Some("u1".to_owned()),
            row: "agent a1".to_owned(),
        }
    }

    #[rstest]
    #[case::untrusted_issuer("https://other", None, Err(JwtDenial::ManagedIssuerUntrusted.into()), &["managed_agent"])]
    #[case::recording_unavailable(
        "https://idp",
        Some(AgentIdentityFailure { policy_unavailable: true, message: "down".to_owned() }),
        Err(JwtDenial::AgentIdentity(AgentIdentityFailure { policy_unavailable: true, message: "down".to_owned() }).into()),
        &["managed_agent", "record_agent agent a1"]
    )]
    #[case::autonomous_skips_every_other_check(
        "https://idp",
        None,
        Ok(Some("a1")),
        &["managed_agent", "record_agent agent a1"]
    )]
    #[tokio::test]
    async fn autonomous_agent(
        #[case] issuer: &str,
        #[case] failure: Option<AgentIdentityFailure>,
        #[case] expected: Result<Option<&str>, Error>,
        #[case] expected_calls: &[&str],
    ) {
        let host = ScriptedHost {
            managed: Some(agent(AgentMode::Autonomous)),
            agent_failure: failure,
            ..host(json!({"iss": issuer, "sub": "someone", "scope": "litellm_proxy_admin"}))
        };
        let identity = authorize_with(&host, request(CHAT), &settings(config()), PROVISIONED).await;
        assert_eq!(
            identity
                .as_ref()
                .map(|identity| identity.agent_id.as_deref()),
            expected.as_ref().map(|id| *id)
        );
        if let Ok(identity) = identity {
            assert_eq!((identity.user_id, identity.is_proxy_admin), (None, false));
        }
        assert_eq!(calls(&host), expected_calls);
    }

    #[tokio::test]
    async fn unavailable_agent_policy_is_a_503() {
        let denial = JwtDenial::AgentIdentity(AgentIdentityFailure {
            policy_unavailable: true,
            message: "down".to_owned(),
        });
        assert_eq!(
            denial.public(),
            PublicDenial::Http {
                status: 503,
                detail: "down".to_owned()
            }
        );
    }

    #[tokio::test]
    async fn delegated_agent_takes_the_granting_team_without_provisioning() {
        let host = ScriptedHost {
            managed: Some(agent(AgentMode::Delegated)),
            delegated_team: Some("t5"),
            teams: HashMap::from([("t5", team("t5"))]),
            users: HashMap::from([("u1", user("u1", "proxy_admin", &[]))]),
            ..host(json!({"iss": "https://idp", "sub": "someone-else"}))
        };
        let identity = authorize_with(&host, request(CHAT), &settings(config()), PROVISIONED)
            .await
            .unwrap();
        assert_eq!(
            (
                identity.user_id.as_deref(),
                identity.team_id.as_deref(),
                identity.is_proxy_admin
            ),
            (Some("u1"), Some("t5"), false)
        );
        assert_eq!(identity.managed_agent, Some(agent(AgentMode::Delegated)));
        assert_eq!(
            calls(&host),
            [
                "managed_agent",
                "record_agent agent a1",
                "delegated a1 team=- explicit=false allowed=-",
                "granting_team t5",
                "resolve_user u1 team=t5",
            ]
        );
    }

    #[rstest]
    #[case::rejected(false, false, Err(JwtDenial::CustomValidate.into()), &["custom_validate"])]
    #[case::accepted(false, true, Ok(Some("u1")), &["custom_validate", "managed_agent", "resolve_user u1 team=-"])]
    #[case::refused_on_a_host_without_operator_code(
        true,
        true,
        Err(JwtDenial::CustomValidateUnavailable.into()),
        &[]
    )]
    #[tokio::test]
    async fn custom_validate_runs_before_anything_else(
        #[case] no_operator_hooks: bool,
        #[case] passes: bool,
        #[case] expected: Result<Option<&str>, Error>,
        #[case] expected_calls: &[&str],
    ) {
        let host = ScriptedHost {
            no_operator_hooks,
            custom_validate_passes: passes,
            users: HashMap::from([("u1", user("u1", "internal_user", &[]))]),
            ..host(json!({"sub": "u1"}))
        };
        let settings = settings(JwtAuthSettings {
            custom_validate: Some("hooks.validate".to_owned()),
            ..config()
        });
        let identity = authorize_with(&host, request(CHAT), &settings, None).await;
        assert_eq!(
            identity
                .as_ref()
                .map(|identity| identity.user_id.as_deref()),
            expected.as_ref().map(|id| *id)
        );
        assert_eq!(calls(&host), expected_calls);
    }

    #[rstest]
    #[case::member_gets_the_rebound_id_and_its_row(
        json!({"sub": "u1"}),
        Ok((Some("canonical"), Some("canonical")))
    )]
    #[case::unknown_user(json!({"sub": "ghost"}), Err(JwtDenial::UserNotFound { user_id: "ghost".to_owned() }.into()))]
    #[case::unknown_admin_keeps_the_claim(
        json!({"sub": "ghost", "scope": "litellm_proxy_admin"}),
        Ok((Some("ghost"), None))
    )]
    #[tokio::test]
    async fn resolve_identity(
        #[case] claims: Value,
        #[case] expected: Result<(Option<&str>, Option<&str>), Error>,
    ) {
        let host = ScriptedHost {
            users: HashMap::from([("u1", user("canonical", "internal_user", &[]))]),
            ..host(claims)
        };
        let identity = crate::credential::jwt::resolve_identity(
            &host,
            &Token::new(TOKEN),
            Timestamp::UNIX_EPOCH,
            &settings(config()),
        )
        .await;
        assert_eq!(
            identity.map(|identity| (identity.user_id, identity.user)),
            expected.map(|(id, user)| (id.map(str::to_owned), user.map(str::to_owned)))
        );
    }

    fn mapping(behavior: UnregisteredJwtClientBehavior) -> JwtSettings {
        settings(JwtAuthSettings {
            virtual_key_claim_field: Some("client_id".into()),
            unregistered_jwt_client_behavior: behavior,
            user_allowed_email_domain: Some("corp.com".to_owned()),
            ..config()
        })
    }

    #[rstest]
    #[case::team_grants_the_route(Some(vec![CHAT.to_owned()]), Ok(Some("t1")))]
    #[case::team_grants_nothing(
        None,
        Err(JwtDenial::Team(TeamDenial::PassthroughRouteDenied { route: CHAT.to_owned(), denied_route: None }).into())
    )]
    #[tokio::test]
    async fn selected_team_must_grant_an_auth_enforced_pass_through_route(
        #[case] allowed: Option<Vec<String>>,
        #[case] expected: Result<Option<&str>, Error>,
    ) {
        let grants = PassthroughGrants {
            allowed,
            denied: vec![],
        };
        let host = ScriptedHost {
            teams: HashMap::from([("t1", team_with("t1", grants))]),
            users: HashMap::from([("u1", user("u1", "internal_user", &["t1"]))]),
            ..host(json!({"sub": "u1", "team": "t1"}))
        };
        let settings = settings(JwtAuthSettings {
            team_id_jwt_field: Some("team".into()),
            ..config()
        });
        let forwarded = [CHAT.to_owned()];
        let request = JwtRequest {
            auth_enforced_pass_through: true,
            forwarded_routes: ForwardedRoutes(&forwarded),
            ..request(CHAT)
        };
        let identity = authorize_with(&host, request, &settings, None).await;
        assert_eq!(
            identity
                .as_ref()
                .map(|identity| identity.team_id.as_deref()),
            expected.as_ref().map(|team| *team)
        );
    }

    #[tokio::test]
    async fn user_over_the_model_budget_is_denied_once_the_token_is_finished() {
        let exceeded = BudgetExceeded {
            current_cost: 2.0,
            max_budget: 1.0,
            message: "Budget has been exceeded!".to_owned(),
            subject: None,
            entity_type: Some("user".to_owned()),
            entity_id: Some("u1".to_owned()),
        };
        let host = ScriptedHost {
            users: HashMap::from([("u1", user("u1", "internal_user", &[]))]),
            jwt_user_over_budget: Some(exceeded.clone()),
            ..host(json!({"sub": "u1"}))
        };
        let Err(denial) = build_with(&host, CHAT, &settings(config())).await else {
            panic!("expected the budget to deny");
        };
        assert_eq!(
            denial.public(),
            Some(PublicDenial::BudgetExceeded(exceeded))
        );
        assert_eq!(
            calls(&host).last().map(String::as_str),
            Some("finish identity")
        );
    }

    async fn build_with(
        host: &ScriptedHost,
        route: &'static str,
        settings: &JwtSettings,
    ) -> Result<JwtOutcome<Labels>, Error> {
        crate::credential::jwt::authenticate(host, &Token::new(TOKEN), &request(route), settings)
            .await
    }

    #[tokio::test]
    async fn auto_register_waits_for_the_policy_to_pass() {
        let host = host(json!({"client_id": "app", "sub": "u1", "email": "u1@evil.com"}));
        let outcome = build_with(
            &host,
            CHAT,
            &mapping(UnregisteredJwtClientBehavior::AutoRegister),
        )
        .await;
        assert!(matches!(
            outcome,
            Err(Error::Jwt(JwtDenial::EmailDomainNotAllowed { .. }))
        ));
        assert_eq!(
            calls(&host),
            [
                "managed_agent",
                "get jwt_key_mapping:client_id:app",
                "mapping client_id=app -",
                "managed_agent"
            ]
        );
    }

    #[tokio::test]
    async fn auto_register_maps_the_authorized_identity() {
        let minted = MappedKey {
            key: Keyed {
                record: KeyRecord {
                    token: Some(KeyHash::new("minted")),
                    ..KeyRecord::default()
                },
                row: "minted row".to_owned(),
            },
            user_email: None,
        };
        let host = ScriptedHost {
            users: HashMap::from([("u1", user("u1", "internal_user", &[]))]),
            keys: HashMap::from([("minted", minted)]),
            ..host(json!({"client_id": "app", "sub": "u1", "email": "u1@corp.com", "org": "o1"}))
        };
        let settings = JwtSettings {
            config: JwtAuthSettings {
                org_id_jwt_field: Some("org".into()),
                ..mapping(UnregisteredJwtClientBehavior::AutoRegister).config
            },
            ..mapping(UnregisteredJwtClientBehavior::AutoRegister)
        };
        assert_eq!(
            build_with(&host, CHAT, &settings).await,
            Ok(JwtOutcome::Finished(
                "finish auto_registered minted row".to_owned()
            ))
        );
        let jwt = host.finished_mapping.lock().unwrap().clone().unwrap();
        assert_eq!(
            jwt.origin,
            KeyOrigin::AutoRegistered {
                minted: true,
                org_id: Some("o1".to_owned()),
            }
        );
        assert_eq!(jwt.user_email.as_deref(), Some("u1@corp.com"));
        assert_eq!(jwt.user.as_deref(), Some("u1"));
        assert_eq!(
            calls(&host),
            [
                "managed_agent",
                "get jwt_key_mapping:client_id:app",
                "mapping client_id=app -",
                "managed_agent",
                "org o1",
                "resolve_user u1 team=-",
                "create_mapping client_id=app existing=-",
                "set jwt_key_mapping:client_id:app 300",
                "key_by_hash minted",
                "finish auto_registered minted row",
            ]
        );
    }

    #[tokio::test]
    async fn auto_registered_admin_caches_the_sentinel_instead() {
        let host = ScriptedHost {
            users: HashMap::from([("u1", user("u1", "internal_user", &[]))]),
            ..host(
                json!({"client_id": "app", "sub": "u1", "email": "u1@corp.com", "scope": "litellm_proxy_admin"}),
            )
        };
        let outcome = build_with(
            &host,
            "/key/info",
            &mapping(UnregisteredJwtClientBehavior::AutoRegister),
        )
        .await;
        assert!(matches!(
            outcome,
            Ok(JwtOutcome::Admin(JwtIdentity {
                is_proxy_admin: true,
                ..
            }))
        ));
        assert_eq!(
            host.json_cache
                .lock()
                .unwrap()
                .get("jwt_key_mapping:client_id:app"),
            Some(&json!(PROXY_ADMIN_SENTINEL))
        );
        assert_eq!(
            calls(&host).last().map(String::as_str),
            Some("set jwt_key_mapping:client_id:app 300")
        );
    }

    #[tokio::test]
    async fn mapped_key_backfills_the_owners_email_and_skips_policy() {
        let key = KeyRecord {
            token: Some(KeyHash::new("h1")),
            user_id: Some("u1".to_owned()),
            ..KeyRecord::default()
        };
        let host = ScriptedHost {
            mappings: HashMap::from([("client_id=app -".to_owned(), "h1".to_owned())]),
            keys: HashMap::from([(
                "h1",
                MappedKey {
                    key: Keyed {
                        record: key.clone(),
                        row: "h1 row".to_owned(),
                    },
                    user_email: None,
                },
            )]),
            users: HashMap::from([(
                "u1",
                User {
                    user_email: Some("owner@corp.com".to_owned()),
                    ..user("u1", "internal_user", &[])
                },
            )]),
            ..host(json!({"client_id": "app", "sub": "u1", "email": "u1@evil.com", "end": ""}))
        };
        let settings = JwtSettings {
            config: JwtAuthSettings {
                custom_validate: Some("hooks.validate".to_owned()),
                end_user_id_jwt_field: Some("end".into()),
                ..mapping(UnregisteredJwtClientBehavior::Reject).config
            },
            ..mapping(UnregisteredJwtClientBehavior::Reject)
        };
        let Ok(JwtOutcome::MappedKey(mapped, jwt)) = build_with(&host, CHAT, &settings).await
        else {
            panic!("expected a mapped key");
        };
        assert_eq!(
            mapped,
            Keyed {
                record: key,
                row: "h1 row".to_owned()
            }
        );
        assert_eq!(
            (jwt.user_email.as_deref(), jwt.end_user_id, jwt.origin),
            (Some("owner@corp.com"), None, KeyOrigin::Mapping)
        );
        assert_eq!(
            calls(&host),
            [
                "managed_agent",
                "get jwt_key_mapping:client_id:app",
                "mapping client_id=app -",
                "set jwt_key_mapping:client_id:app 300",
                "key_by_hash h1",
                "user u1 u1@evil.com",
            ]
        );
    }

    #[tokio::test]
    async fn managed_agents_cannot_use_a_mapping() {
        let host = ScriptedHost {
            managed: Some(agent(AgentMode::Delegated)),
            json_cache: HashMap::from([(
                "jwt_key_mapping:client_id:app".to_owned(),
                json!(NO_MAPPING_SENTINEL),
            )])
            .into(),
            ..host(json!({"client_id": "app"}))
        };
        let outcome = build_with(
            &host,
            CHAT,
            &mapping(UnregisteredJwtClientBehavior::FallbackTeamMapping),
        )
        .await;
        assert_eq!(outcome, Err(JwtDenial::ManagedAgentNeedsDirectJwt.into()));
        assert_eq!(calls(&host), ["managed_agent"]);
    }

    #[tokio::test]
    async fn scim_deactivated_user_is_denied_after_policy() {
        let host = ScriptedHost {
            users: HashMap::from([(
                "u1",
                User {
                    deactivated: true,
                    ..user("u1", "internal_user", &[])
                },
            )]),
            ..host(json!({"sub": "u1"}))
        };
        let outcome = build_with(&host, CHAT, &settings(config())).await;
        assert_eq!(
            outcome,
            Err(JwtDenial::UserDeactivated {
                user_id: Some("u1".to_owned()),
            }
            .into())
        );
        assert_eq!(calls(&host), ["managed_agent", "resolve_user u1 team=-"]);
    }

    fn team_id_of(identity: &Result<JwtIdentity<Labels>, Error>) -> Result<Option<&str>, &Error> {
        identity
            .as_ref()
            .map(|identity| identity.team_id.as_deref())
    }

    /// A token whose team comes from its role: the `Team` role by mapping, the team in `oid`.
    fn rbac_team(config: JwtAuthSettings) -> JwtAuthSettings {
        JwtAuthSettings {
            roles_jwt_field: Some("roles".into()),
            role_mappings: Some(Box::new([RoleMapping {
                role: "team-role".to_owned(),
                internal_role: RbacRole::Team,
            }])),
            object_id_jwt_field: Some("oid".into()),
            ..config
        }
    }

    fn rbac_team_claims() -> Value {
        json!({"roles": ["team-role"], "oid": "t1"})
    }

    #[tokio::test]
    async fn a_jwt_is_verified_even_when_userinfo_is_enabled() {
        let host = host(json!({"sub": "u1", "scope": "litellm_proxy_admin"}));
        let denial = crate::credential::jwt::authorize_jwt(
            &host,
            &Token::new("a.b.c"),
            &request(CHAT),
            &settings(config()),
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(
            denial,
            Error::Jwt(JwtDenial::Verify(VerifyDenial::Malformed(_)))
        ));
        assert_eq!(host.calls(), [] as [&str; 0]);
    }

    #[rstest]
    #[case::mapped_model(true, "gpt-a", Ok(Some("u1")), &["managed_agent", "resolve_user u1 team=-"])]
    #[case::unmapped_model_is_denied_before_the_user_loads(
        true,
        "gpt-b",
        Err(JwtDenial::Rbac(RbacDenial::ScopeModel {
            model: "gpt-b".to_owned(),
            allowed: vec!["gpt-a".to_owned()],
        }).into()),
        &["managed_agent"]
    )]
    #[case::not_enforced(false, "gpt-b", Ok(Some("u1")), &["managed_agent", "resolve_user u1 team=-"])]
    #[tokio::test]
    async fn scope_mappings_gate_the_model(
        #[case] enforce_scope_based_access: bool,
        #[case] model: &'static str,
        #[case] expected: Result<Option<&str>, Error>,
        #[case] expected_calls: &[&str],
    ) {
        let host = ScriptedHost {
            users: HashMap::from([("u1", user("u1", "internal_user", &[]))]),
            ..host(json!({"sub": "u1", "scope": "read"}))
        };
        let settings = settings(JwtAuthSettings {
            enforce_scope_based_access,
            scope_mappings: Some(Box::new([ScopeMapping {
                scope: "read".to_owned(),
                models: Some(Box::new(["gpt-a".to_owned()])),
                routes: None,
            }])),
            ..config()
        });
        let request = JwtRequest {
            model: Some(model),
            ..request(CHAT)
        };
        let identity = authorize_with(&host, request, &settings, None).await;
        assert_eq!(
            identity
                .as_ref()
                .map(|identity| identity.user_id.as_deref()),
            expected.as_ref().map(|user| *user)
        );
        assert_eq!(calls(&host), expected_calls);
    }

    #[rstest]
    #[case::member_team_outside_the_token(
        json!({"sub": "u1", "groups": ["t1"]}),
        None,
        "t2",
        CHAT,
        Ok(Some("t2"))
    )]
    #[case::existing_team_the_user_is_not_in(
        json!({"sub": "u1", "groups": ["t1"]}),
        None,
        "t9",
        CHAT,
        Err(JwtDenial::Team(TeamDenial::HeaderTeamNotMember { header_value: "t9".to_owned() }).into())
    )]
    #[case::route_outside_team_allowed_routes(
        json!({"sub": "u1", "groups": ["t1"]}),
        None,
        "t2",
        "/key/generate",
        Err(JwtDenial::Team(TeamDenial::HeaderTeamRouteNotAllowed {
            header_value: "t2".to_owned(),
            route: "/key/generate".to_owned(),
        }).into())
    )]
    #[case::default_team_is_not_a_claimed_team(
        json!({"sub": "u1"}),
        Some("t9"),
        "t9",
        CHAT,
        Err(JwtDenial::Team(TeamDenial::HeaderTeamNotMember { header_value: "t9".to_owned() }).into())
    )]
    #[tokio::test]
    async fn header_team_outside_the_token_must_be_a_db_membership_under_db_fallback(
        #[case] claims: Value,
        #[case] team_id_default: Option<&str>,
        #[case] header: &'static str,
        #[case] route: &'static str,
        #[case] expected: Result<Option<&str>, Error>,
    ) {
        let host = ScriptedHost {
            teams: HashMap::from([("t1", team("t1")), ("t2", team("t2")), ("t9", team("t9"))]),
            users: HashMap::from([("u1", user("u1", "internal_user", &["t2"]))]),
            ..host(claims)
        };
        let settings = settings(JwtAuthSettings {
            fallback_to_db_teams: true,
            team_ids_jwt_field: Some("groups".into()),
            team_id_default: team_id_default.map(str::to_owned),
            ..config()
        });
        let request = JwtRequest {
            team_header: Some(header),
            ..request(route)
        };
        let identity = authorize_with(&host, request, &settings, None).await;
        assert_eq!(team_id_of(&identity), expected.as_ref().map(|team| *team));
        assert_eq!(
            calls(&host),
            [
                "managed_agent".to_owned(),
                format!("team {header}"),
                format!("team {header}"),
                format!("resolve_user u1 team={header}"),
            ]
        );
    }

    #[rstest]
    #[case::binds_without_a_db_membership(None, Ok(Some("t1")), &["managed_agent"])]
    #[case::header_cannot_swap_it(
        Some("t2"),
        Err(JwtDenial::Team(TeamDenial::HeaderTeamNotAllowed {
            header_value: "t2".to_owned(),
            allowed_team_ids: BTreeSet::new(),
        }).into()),
        &["managed_agent", "alias t2"]
    )]
    #[tokio::test]
    async fn rbac_team_is_pinned_under_db_fallback(
        #[case] header: Option<&'static str>,
        #[case] expected: Result<Option<&str>, Error>,
        #[case] expected_calls: &[&str],
    ) {
        let host = ScriptedHost {
            teams: HashMap::from([("t1", team("t1")), ("t2", team("t2"))]),
            ..host(rbac_team_claims())
        };
        let settings = settings(rbac_team(JwtAuthSettings {
            fallback_to_db_teams: true,
            ..config()
        }));
        let request = JwtRequest {
            team_header: header,
            ..request(CHAT)
        };
        let identity = authorize_with(&host, request, &settings, None).await;
        assert_eq!(team_id_of(&identity), expected.as_ref().map(|team| *team));
        assert_eq!(calls(&host), expected_calls);
    }

    #[rstest]
    #[case::default_team_is_ignored(
        JwtAuthSettings { team_id_default: Some("tdef".to_owned()), ..config() },
        json!({"sub": "u1"}),
        CHAT,
        Some("t1"),
        &["managed_agent", "resolve_user u1 team=-", "team t1", "membership u1 t1"]
    )]
    #[case::claimed_team_binds(
        JwtAuthSettings { team_id_jwt_field: Some("team".into()), ..config() },
        json!({"sub": "u1", "team": "t2"}),
        CHAT,
        Some("t2"),
        &["managed_agent", "team t2", "resolve_user u1 team=t2"]
    )]
    #[case::aliased_team_binds(
        JwtAuthSettings { team_alias_jwt_field: Some("team_name".into()), ..config() },
        json!({"sub": "u1", "team_name": "eng"}),
        CHAT,
        Some("t3"),
        &["managed_agent", "alias eng", "resolve_user u1 team=t3"]
    )]
    #[case::route_outside_team_allowed_routes_picks_no_team(
        JwtAuthSettings { team_allowed_routes: Box::new(["openai_routes".to_owned()]), ..config() },
        json!({"sub": "u1"}),
        "/key/info",
        None,
        &["managed_agent", "resolve_user u1 team=-", "team t1"]
    )]
    #[tokio::test]
    async fn db_team_fallback_is_only_for_a_token_without_a_team(
        #[case] config: JwtAuthSettings,
        #[case] claims: Value,
        #[case] route: &'static str,
        #[case] team_id: Option<&str>,
        #[case] expected_calls: &[&str],
    ) {
        let aliased = Team {
            team_id: "t3".to_owned(),
            models: Some(Vec::new()),
            row: "t3".to_owned(),
            ..Team::default()
        };
        let host = ScriptedHost {
            teams: HashMap::from([
                ("t1", team("t1")),
                ("t2", team("t2")),
                ("tdef", team("tdef")),
            ]),
            aliases: HashMap::from([("eng", Ok(AliasRead::Found(aliased)))]),
            users: HashMap::from([("u1", user("u1", "internal_user", &["t1"]))]),
            ..host(claims)
        };
        let settings = settings(JwtAuthSettings {
            fallback_to_db_teams: true,
            ..config
        });
        let identity = authorize_with(&host, request(route), &settings, None)
            .await
            .unwrap();
        assert_eq!(identity.team_id.as_deref(), team_id);
        assert_eq!(calls(&host), expected_calls);
    }

    #[rstest]
    #[case::header_team(
        JwtAuthSettings { team_ids_jwt_field: Some("groups".into()), ..config() },
        json!({"sub": "u1", "groups": ["t1"]}),
        Some("t1")
    )]
    #[case::rbac_team(rbac_team(config()), rbac_team_claims(), None)]
    #[case::db_fallback_team(
        JwtAuthSettings { fallback_to_db_teams: true, ..config() },
        json!({"sub": "u1"}),
        None
    )]
    #[tokio::test]
    async fn every_team_source_must_grant_an_auth_enforced_pass_through_route(
        #[case] config: JwtAuthSettings,
        #[case] claims: Value,
        #[case] header: Option<&'static str>,
        #[values(true, false)] granted: bool,
    ) {
        let grants = PassthroughGrants {
            allowed: granted.then(|| vec![CHAT.to_owned()]),
            denied: vec![],
        };
        let host = ScriptedHost {
            teams: HashMap::from([("t1", team_with("t1", grants))]),
            users: HashMap::from([("u1", user("u1", "internal_user", &["t1"]))]),
            ..host(claims)
        };
        let forwarded = [CHAT.to_owned()];
        let request = JwtRequest {
            team_header: header,
            auth_enforced_pass_through: true,
            forwarded_routes: ForwardedRoutes(&forwarded),
            ..request(CHAT)
        };
        let identity = authorize_with(&host, request, &settings(config), None).await;
        let denied: Error = JwtDenial::Team(TeamDenial::PassthroughRouteDenied {
            route: CHAT.to_owned(),
            denied_route: None,
        })
        .into();
        assert_eq!(
            team_id_of(&identity),
            if granted {
                Ok(Some("t1"))
            } else {
                Err(&denied)
            }
        );
    }

    const PLAIN_AGENT: RegisteredAgent = RegisteredAgent {
        agent_id: String::new(),
        identity_managed: false,
        has_identity: false,
        has_legacy_identity: false,
    };

    fn agent_settings() -> JwtSettings {
        settings(JwtAuthSettings {
            agent_id_jwt_field: Some("agent".into()),
            admin_allowed_routes: Box::new(["info_routes".to_owned()]),
            ..config()
        })
    }

    fn unregistered_agent() -> Error {
        JwtDenial::UnregisteredAgent {
            agent_id_jwt_field: Some("agent".into()),
            claim: "bot".to_owned(),
        }
        .into()
    }

    #[rstest]
    #[case::registered(Some(PLAIN_AGENT), Ok(Some("a-1")))]
    #[case::unregistered(None, Err(unregistered_agent()))]
    #[case::identity_managed(
        Some(RegisteredAgent { identity_managed: true, ..PLAIN_AGENT }),
        Err(unregistered_agent())
    )]
    #[case::bound_identity(
        Some(RegisteredAgent { has_identity: true, ..PLAIN_AGENT }),
        Err(unregistered_agent())
    )]
    #[case::legacy_identity(
        Some(RegisteredAgent { has_legacy_identity: true, ..PLAIN_AGENT }),
        Err(unregistered_agent())
    )]
    #[tokio::test]
    async fn agent_claim_only_names_an_agent_without_its_own_identity(
        #[case] registered: Option<RegisteredAgent>,
        #[case] expected: Result<Option<&str>, Error>,
        #[values("", "litellm_proxy_admin")] scope: &str,
    ) {
        let host = ScriptedHost {
            agents: registered
                .map(|agent| {
                    let agent = RegisteredAgent {
                        agent_id: "a-1".to_owned(),
                        ..agent
                    };
                    ("bot", agent)
                })
                .into_iter()
                .collect(),
            users: HashMap::from([("u1", user("u1", "internal_user", &[]))]),
            ..host(json!({"sub": "u1", "agent": "bot", "scope": scope}))
        };
        let identity = authorize_with(&host, request("/key/info"), &agent_settings(), None).await;
        assert_eq!(
            identity
                .as_ref()
                .map(|identity| identity.agent_id.as_deref()),
            expected.as_ref().map(|agent| *agent)
        );
        assert_eq!(
            calls(&host)[..2],
            ["managed_agent", "agent bot"],
            "the agent is resolved before any user read, admin or not"
        );
        assert_eq!(calls(&host).len() == 2, expected.is_err());
    }

    #[tokio::test]
    async fn org_alias_and_end_user_claims_reach_the_identity() {
        let host = ScriptedHost {
            orgs: HashMap::from([(
                "acme",
                Org {
                    organization_id: "o1".to_owned(),
                    row: "org o1".to_owned(),
                },
            )]),
            users: HashMap::from([("u1", user("u1", "internal_user", &[]))]),
            ..host(json!({"sub": "u1", "org_name": "acme", "end": "e1"}))
        };
        let settings = settings(JwtAuthSettings {
            org_alias_jwt_field: Some("org_name".into()),
            end_user_id_jwt_field: Some("end".into()),
            ..config()
        });
        let identity = authorize_with(&host, request(CHAT), &settings, None)
            .await
            .unwrap();
        assert_eq!(
            (identity.org_id.as_deref(), identity.end_user_id.as_deref()),
            (Some("o1"), Some("e1"))
        );
        assert_eq!(
            identity.rows,
            JwtRows {
                user: Some("u1".to_owned()),
                org: Some("org o1".to_owned()),
                end_user: Some("end-user e1".to_owned()),
                ..JwtRows::default()
            }
        );
        assert_eq!(
            calls(&host),
            [
                "managed_agent",
                "org_alias acme",
                "resolve_user u1 team=-",
                "end_user e1",
            ]
        );
    }
}
