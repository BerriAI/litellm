use std::collections::BTreeSet;

use litellm_config::{LitellmRole, RbacRole};
use serde_json::{Map, Value};

use super::{
    FinishedJwt, JwtDenial, JwtIdentity, JwtOutcome, JwtRequest, JwtRows, JwtSettings, TeamDenial,
    claims::{self, UserInfo},
    mapping::{self, AfterAuthorize, BeforeAuthorize},
    rbac,
    team::{self, Team, TeamClaims, TeamRead, TeamSettings},
    verify::{self, Claims},
};
use crate::{
    AuthHost, Error, Rows,
    host::{AgentRow, MembershipRow, UserRow},
    passthrough::PassthroughRequest,
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

fn non_empty(text: &&str) -> bool {
    !text.is_empty()
}

pub(super) fn filled(text: Option<String>) -> Option<String> {
    text.filter(|text| !text.is_empty())
}

/// `auth_jwt`, or `get_oidc_userinfo` for a token that is not a JWT when UserInfo is enabled.
async fn decode(
    host: &impl AuthHost,
    token: &str,
    settings: &JwtSettings,
    now: f64,
) -> Result<Claims, Error> {
    if settings.config.oidc_userinfo_enabled && !verify::is_jwt(token) {
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
    if settings.config.custom_validate.is_some()
        && !host
            .custom_validate(&Value::Object(claims.0.clone()))
            .await?
    {
        return Err(JwtDenial::CustomValidate.into());
    }
    Ok(())
}

/// `JWTAuthManager.authenticate_jwt`
pub(super) async fn authenticate_token(
    host: &impl AuthHost,
    token: &str,
    settings: &JwtSettings,
    now: f64,
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
            agent_id_jwt_field: settings.config.agent_id_jwt_field.clone(),
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
    let Some(user_id) = user_id.filter(non_empty) else {
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
    match read.user {
        UserRead::Found(user) => {
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
        UserRead::Absent => Err(JwtDenial::UserNotFound {
            user_id: user_id.to_owned(),
        }
        .into()),
    }
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
        Some(RbacRole::InternalUser) => {
            filled(claims::object_id(claims, settings)).or(info.user_id)
        }
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

/// A `get_team_object` whose raise `authorize_jwt` lets through.
fn found<R>(read: TeamRead<R>, team_id: &str) -> Result<Team<R>, TeamDenial> {
    match read {
        TeamRead::Found(team) => Ok(team),
        TeamRead::Absent | TeamRead::Unreadable => Err(TeamDenial::TeamNotFound {
            team_id: team_id.to_owned(),
        }),
    }
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
        if !verify::managed_issuer_is_trusted(claims.0.get("iss").unwrap_or(&Value::Null), settings)
        {
            return Err(JwtDenial::ManagedIssuerUntrusted.into());
        }
        if !crate::routes::managed_agent_route_allowed(route, request.method) {
            return Err(JwtDenial::ManagedRouteNotAllowed.into());
        }
        if let Some(failure) = host.record_agent_authentication(&agent.row).await? {
            return Err(JwtDenial::AgentIdentity(failure).into());
        }
        if agent.mode == AgentMode::Autonomous {
            let identity = JwtIdentity {
                claims: claims_value,
                is_proxy_admin: false,
                user_id: None,
                user_email: None,
                team_id: None,
                org_id: None,
                end_user_id: None,
                agent_id: Some(agent.agent_id.clone()),
                managed_agent: managed,
                rows: JwtRows::default(),
            };
            return Ok(Authorized {
                identity,
                user: None,
            });
        }
    }
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
    if settings.config.enforce_scope_based_access
        && settings
            .config
            .scope_mappings
            .as_deref()
            .is_some_and(|mappings| !mappings.is_empty())
    {
        rbac::check_scope_based_access(&scopes, model, settings)?;
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
    let object_id = filled(claims::object_id(claims, settings));
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
        let user = match (provisioning, user_id.as_deref()) {
            (None, _) => claim_identity(host, claims, settings).await?.1,
            (Some(_), _) if !settings.has_db => None,
            (Some(_), None) => None,
            (Some(_), Some(id)) => {
                let lookup = UserLookup {
                    user_id: id,
                    user_email: user_email.as_deref(),
                    sso_user_id: Some(id),
                    upsert: false,
                };
                match host.user(&lookup).await? {
                    UserRead::Found(user) => Some(user),
                    UserRead::Absent => None,
                }
            }
        };
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

    let claim_team_ids = claims::all_team_ids(claims, settings);
    let team_alias = filled(claims::team_alias(claims, settings));
    let specific_team_id = claims::team_id(claims, settings);
    let db_team_fallback = settings.config.fallback_to_db_teams
        && claim_team_ids.is_empty()
        && team_alias.is_none()
        && rbac_team_id.is_none();
    let team_ids: BTreeSet<String> = claims::team_ids(claims, settings)
        .into_iter()
        .chain(filled(specific_team_id.clone()).filter(|_| !db_team_fallback))
        .collect();
    let header_db_fallback = settings.config.fallback_to_db_teams && rbac_team_id.is_none();
    let header_team =
        team::header_team(host, request.team_header, &team_ids, header_db_fallback).await?;
    let passthrough = PassthroughRequest {
        route,
        auth_enforced: request.auth_enforced_pass_through,
        forwarded_routes: request.forwarded_routes,
        jwt_team_allowed_routes: &settings.config.team_allowed_routes,
    };

    let (team_id, loaded_team) = match &header_team {
        Some(header_team) => {
            let loaded = team::load_header_team(host, header_team, team_id_upsert).await?;
            (Some(header_team.team_id.clone()), Some(loaded))
        }
        None if rbac_team_id.is_none() && !db_team_fallback => {
            let team_claims = TeamClaims {
                team_id: specific_team_id.as_deref(),
                has_team_id_claim: !claim_team_ids.is_empty(),
                team_alias: team_alias.as_deref(),
                team_id_jwt_field: settings.config.team_id_jwt_field.as_deref(),
                team_alias_jwt_field: settings.config.team_alias_jwt_field.as_deref(),
            };
            let found =
                team::specific_team(host, team_claims, team_id_upsert, team_settings).await?;
            (found.as_ref().map(|team| team.team_id.clone()), found)
        }
        None => (rbac_team_id, None),
    };
    let (team_id, loaded_team) = match team_id {
        None => {
            let route_allowed = rbac::allowed_routes_check(RbacRole::Team, route, settings);
            let found = team::select_team(
                host,
                &team_ids,
                model,
                &passthrough,
                route_allowed,
                team_settings,
            )
            .await?;
            (found.as_ref().map(|team| team.team_id.clone()), found)
        }
        Some(team_id) => (Some(team_id), loaded_team),
    };
    let loaded_team = match team_id.as_deref() {
        Some(id) if loaded_team.is_none() && request.auth_enforced_pass_through => {
            Some(found(host.team(id, team_id_upsert).await?, id)?)
        }
        _ => loaded_team,
    };

    let granting_team = match &managed {
        None => team_id.clone(),
        Some(agent) => {
            let scoped: Option<BTreeSet<String>> = match (&team_id, &team_alias) {
                _ if !claim_team_ids.is_empty() => Some(claim_team_ids.iter().cloned().collect()),
                (Some(team_id), Some(_)) => Some(BTreeSet::from([team_id.clone()])),
                _ => None,
            };
            let delegated = DelegatedTeam {
                user_id: agent.user_id.as_deref(),
                agent_id: &agent.agent_id,
                team_id: team_id.as_deref(),
                explicit_team: header_team.is_some(),
                allowed_team_ids: scoped
                    .as_ref()
                    .filter(|_| !settings.config.fallback_to_db_teams),
            };
            host.delegated_agent_team(&delegated).await?
        }
    };
    let team_route_allowed =
        rbac::is_team_route_allowed(route, request.auth_enforced_pass_through, settings);
    if granting_team.is_some() && granting_team != team_id && !team_route_allowed {
        return Err(JwtDenial::GrantingTeamRouteNotAllowed.into());
    }
    let selected_team_id = granting_team.or_else(|| team_id.clone());
    let selected_team = match selected_team_id.as_deref() {
        Some(id) if selected_team_id != team_id => Some(found(host.granting_team(id).await?, id)?),
        _ => loaded_team,
    };
    team::passthrough_gate(
        selected_team_id.as_deref(),
        selected_team.as_ref(),
        &passthrough,
    )?;

    let org = match (
        org_id.as_deref().filter(non_empty),
        filled(claims::org_alias(claims, settings)),
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
    let end_user = match end_user_id.as_deref().filter(non_empty) {
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
        None if db_team_fallback => {
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
            if let Some(header_team) = header_team
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
            .and_then(|user| filled(user.user_email.clone()))
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

/// The team a fallback joined and the membership it loaded with it.
fn joined<R: Rows>(
    found: Option<team::Joined<R>>,
) -> (Option<String>, Option<Team<R::Team>>, Option<R::Membership>) {
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
    token: &str,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
) -> Result<JwtOutcome<H::Rows>, Error> {
    let claims = decode(host, token, settings, request.now).await?;
    let pending = match mapping::before_authorize(host, &claims, settings).await? {
        BeforeAuthorize::Mapped(outcome) => return Ok(outcome),
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

    use litellm_config::{JwtAuthSettings, JwtLitellmRoleMap, UnregisteredJwtClientBehavior};
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::{
        BudgetExceeded, ErrorType, PublicDenial,
        jwt::AgentIdentityFailure,
        jwt::{
            RbacDenial, Team,
            mapping::{KeyOrigin, MappedKey, NO_MAPPING_SENTINEL, PROXY_ADMIN_SENTINEL},
        },
        key::{KeyRecord, Keyed},
        passthrough::PassthroughGrants,
        testing::{Labels, ScriptedHost},
    };

    const CHAT: &str = "/chat/completions";
    const TOKEN: &str = "opaque";

    fn config() -> JwtAuthSettings {
        JwtAuthSettings {
            oidc_userinfo_enabled: true,
            oidc_userinfo_endpoint: Some("https://idp/userinfo".to_owned()),
            user_id_jwt_field: Some("sub".to_owned()),
            user_email_jwt_field: Some("email".to_owned()),
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
            ..JwtSettings::default()
        }
    }

    fn request(route: &'static str) -> JwtRequest<'static> {
        JwtRequest {
            route,
            method: Some("POST"),
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
        crate::jwt::authorize_jwt(host, TOKEN, &request, settings, provisioning).await
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
            team_id_jwt_field: Some("team".to_owned()),
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
            team_ids_jwt_field: Some("groups".to_owned()),
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
            org_id_jwt_field: Some("org".to_owned()),
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
            team_ids_jwt_field: Some("groups".to_owned()),
            team_id_jwt_field: Some("team".to_owned()),
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
            roles_jwt_field: Some("roles".to_owned()),
            jwt_litellm_role_map: Some(Box::new([JwtLitellmRoleMap {
                jwt_role: "admin*".to_owned(),
                litellm_role: LitellmRole::ProxyAdmin,
            }])),
            team_ids_jwt_field: Some("groups".to_owned()),
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
    #[case::rejected(false, Err(JwtDenial::CustomValidate.into()), &["custom_validate"])]
    #[case::accepted(true, Ok(Some("u1")), &["custom_validate", "managed_agent", "resolve_user u1 team=-"])]
    #[tokio::test]
    async fn custom_validate_runs_before_anything_else(
        #[case] passes: bool,
        #[case] expected: Result<Option<&str>, Error>,
        #[case] expected_calls: &[&str],
    ) {
        let host = ScriptedHost {
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
        let identity = crate::jwt::resolve_identity(&host, TOKEN, 0.0, &settings(config())).await;
        assert_eq!(
            identity.map(|identity| (identity.user_id, identity.user)),
            expected.map(|(id, user)| (id.map(str::to_owned), user.map(str::to_owned)))
        );
    }

    fn mapping(behavior: UnregisteredJwtClientBehavior) -> JwtSettings {
        settings(JwtAuthSettings {
            virtual_key_claim_field: Some("client_id".to_owned()),
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
            team_id_jwt_field: Some("team".to_owned()),
            ..config()
        });
        let forwarded = [CHAT.to_owned()];
        let request = JwtRequest {
            auth_enforced_pass_through: true,
            forwarded_routes: &forwarded,
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
        crate::jwt::authenticate(host, TOKEN, &request(route), settings).await
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
                    token: Some("minted".to_owned()),
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
                org_id_jwt_field: Some("org".to_owned()),
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
            token: Some("h1".to_owned()),
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
                end_user_id_jwt_field: Some("end".to_owned()),
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
}
