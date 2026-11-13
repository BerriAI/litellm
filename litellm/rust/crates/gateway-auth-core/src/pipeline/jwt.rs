use jiff::Timestamp;
use litellm_config::{LitellmRole, RbacRole, UnregisteredJwtClientBehavior as Behavior};
use serde_json::Value;

use crate::{
    AuthHost, Error, JwtDenial, Rows, Token,
    authn::jwt::{
        self, Claims, JwtRequest, JwtSettings, KeyOrigin, MappedJwt,
        mapping::{self, Mapping, Pending, Resolved},
    },
    authz::{RoutePolicy, Target},
    commit::{self, Grant, Grants},
    host::{KeyRow, UserRow},
    identity::{
        Caller, CallerKind, Principal,
        jwt::{
            AgentMode, ClaimIdentity, FinishedJwt, JwtIdentity, JwtMapping, JwtOutcome, JwtRows,
            Provisioning, Team, TeamSettings, UserInfo, agent, caller, claims, org,
            team::{self, Joined},
            user::{self, ResolvedUser},
        },
    },
};

/// An identity and the grants it earned, which `commit` writes once nothing refuses the request.
struct Authorized<R: Rows> {
    identity: JwtIdentity<R>,
    grants: Grants<R>,
}

/// The builder's JWT block: the token, the key it maps to, the operator's check, then who it
/// acts as and what it may call, then the grants and the token the host finishes.
pub(crate) async fn build<H: AuthHost>(
    host: &H,
    token: &Token,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
    policy: &RoutePolicy<'_>,
) -> Result<JwtOutcome<H::Rows>, Error> {
    let claims = jwt::decode(host, token, settings, request.now).await?;
    let pending = match mapping::mapped_key(host, &claims, settings).await? {
        Mapping::Mapped(outcome) => return Ok(*outcome),
        Mapping::Pending(pending) => Some(pending),
        Mapping::Unmapped => None,
    };
    jwt::custom_validate(host, &claims, settings).await?;
    let provisioning = Some(Provisioning::from(settings));
    let authorized = authorize(host, &claims, request, settings, policy, provisioning).await?;
    match pending {
        Some(pending) => register(host, authorized, &pending, settings).await,
        None => {
            commit::apply(host, authorized.grants).await?;
            finished(host, authorized.identity).await
        }
    }
}

/// `JWTAuthManager.authorize_jwt`. `auth_builder` passes the configured provisioning; the MCP
/// token flows pass `None`.
pub(crate) async fn authorize_jwt<H: AuthHost>(
    host: &H,
    token: &Token,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
    policy: &RoutePolicy<'_>,
    provisioning: Option<Provisioning>,
) -> Result<JwtIdentity<H::Rows>, Error> {
    let claims = jwt::decode(host, token, settings, request.now).await?;
    jwt::custom_validate(host, &claims, settings).await?;
    let Authorized { identity, grants } =
        authorize(host, &claims, request, settings, policy, provisioning).await?;
    commit::apply(host, grants).await?;
    Ok(identity)
}

/// `JWTAuthManager.authenticate_jwt`, then `_resolve_jwt_to_virtual_key` without provisioning.
pub(crate) async fn resolve_mapping<H: AuthHost>(
    host: &H,
    token: &Token,
    now: Timestamp,
    settings: &JwtSettings,
) -> Result<JwtMapping<KeyRow<H>>, Error> {
    let claims = jwt::decode(host, token, settings, now).await?;
    jwt::custom_validate(host, &claims, settings).await?;
    Ok(match mapping::resolve(host, &claims, settings).await? {
        Resolved::Key(key) => JwtMapping::Key(Box::new(key)),
        Resolved::Pending(_) => JwtMapping::Pending,
        Resolved::Unmapped => JwtMapping::Unmapped,
    })
}

/// `JWTAuthManager.resolve_identity`
pub(crate) async fn resolve_identity<H: AuthHost>(
    host: &H,
    token: &Token,
    now: Timestamp,
    settings: &JwtSettings,
    policy: &RoutePolicy<'_>,
) -> Result<ClaimIdentity<UserRow<H>>, Error> {
    let claims = jwt::decode(host, token, settings, now).await?;
    jwt::custom_validate(host, &claims, settings).await?;
    policy.claimed_email(&claims::user_info(&claims, settings))?;
    let (identity, _) = user::claim_identity(host, &claims, settings).await?;
    Ok(identity)
}

/// What a caller claims about itself before any read: the user id an RBAC role rebinds to its
/// object id, and the team an RBAC team token names.
fn claimed_ids<R: Rows>(
    caller: &Caller<R>,
    info: &UserInfo,
    claims: &Claims,
    settings: &JwtSettings,
) -> (Option<String>, Option<String>) {
    let object_id = claims::object_id(claims, settings).filter(|text| !text.is_empty());
    match (caller.rbac_role, object_id) {
        (Some(RbacRole::Team), Some(object_id)) => (Some(object_id), info.user_id.clone()),
        (Some(RbacRole::InternalUser), Some(object_id)) => (None, Some(object_id)),
        _ => (None, info.user_id.clone()),
    }
}

/// `JWTAuthManager.authorize_jwt` after `authenticate_jwt`: the caller and what it may call, then
/// a proxy admin's or an autonomous agent's identity, or the team, organization and user behind
/// anyone else, each check as soon as what it reads exists.
async fn authorize<H: AuthHost>(
    host: &H,
    claims: &Claims,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
    policy: &RoutePolicy<'_>,
    provisioning: Option<Provisioning>,
) -> Result<Authorized<H::Rows>, Error> {
    let target = Target::from(request);
    let caller = caller::caller(host, claims, settings).await?;
    policy.credential(&caller, &target)?;
    let info = match &caller.agent {
        Some(agent) => UserInfo {
            user_id: agent.user_id.clone(),
            ..UserInfo::default()
        },
        None => claims::user_info(claims, settings),
    };
    policy.email_domain(&caller, &info)?;
    let claims_value = Value::Object(claims.0.clone());
    let grants = match &caller.agent {
        Some(agent) => Grants::default().with(Grant::RecordAgentAuthentication(agent.row.clone())),
        None => Grants::default(),
    };
    if let (CallerKind::ManagedAgent(AgentMode::Autonomous), Some(agent)) =
        (caller.kind, caller.agent.clone())
    {
        let identity = agent::autonomous_identity::<H::Rows>(claims_value, agent);
        return Ok(Authorized { identity, grants });
    }
    if caller.kind == CallerKind::ProxyAdmin {
        let identity = admin(
            host,
            claims,
            caller,
            info,
            request,
            settings,
            policy,
            provisioning,
        )
        .await?;
        return Ok(Authorized { identity, grants });
    }
    let principal = principal(
        host,
        claims,
        caller,
        info,
        request,
        settings,
        policy,
        provisioning,
        grants,
    )
    .await?;
    policy.principal(&principal, &target)?;
    Ok(identity_of(claims_value, principal))
}

/// A proxy admin's identity: the header team on LLM routes and the admin's user row.
#[expect(
    clippy::too_many_arguments,
    reason = "each stage input is read once, where the admin branch of `authorize_jwt` reads it"
)]
async fn admin<H: AuthHost>(
    host: &H,
    claims: &Claims,
    caller: Caller<H::Rows>,
    info: UserInfo,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
    policy: &RoutePolicy<'_>,
    provisioning: Option<Provisioning>,
) -> Result<JwtIdentity<H::Rows>, Error> {
    if provisioning.is_none() {
        policy.claimed_email(&info)?;
    }
    let (_, user_id) = claimed_ids(&caller, &info, claims, settings);
    let team_id_upsert = provisioning.is_some_and(|provisioning| provisioning.team_id_upsert);
    let team = team::admin_header_team(
        host,
        request.team_header,
        request.is_llm_api_route,
        team_id_upsert,
    )
    .await;
    let user = user::admin_user(
        host,
        claims,
        settings,
        provisioning,
        user_id.as_deref(),
        info.user_email.as_deref(),
    )
    .await?;
    if user.as_ref().is_some_and(|user| user.deactivated) {
        return Err(JwtDenial::UserDeactivated { user_id }.into());
    }
    Ok(JwtIdentity {
        claims: Value::Object(claims.0.clone()),
        is_proxy_admin: true,
        user_id,
        user_email: info.user_email,
        team_id: team.as_ref().map(|team| team.team_id.clone()),
        org_id: claims::org_id(claims, settings),
        end_user_id: None,
        agent_id: caller.agent_id,
        managed_agent: None,
        rows: JwtRows {
            user: user.map(|user| user.row),
            team: team.map(|team| team.row),
            ..JwtRows::default()
        },
    })
}

/// The team, organization, user and end user behind a caller that is neither a proxy admin nor
/// an autonomous agent, with the grants it earns.
#[expect(
    clippy::too_many_arguments,
    reason = "each stage input is read once, where `authorize_jwt` reads it"
)]
async fn principal<H: AuthHost>(
    host: &H,
    claims: &Claims,
    caller: Caller<H::Rows>,
    info: UserInfo,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
    policy: &RoutePolicy<'_>,
    provisioning: Option<Provisioning>,
    grants: Grants<H::Rows>,
) -> Result<Principal<H::Rows>, Error> {
    let target = Target::from(request);
    let (rbac_team_id, user_id) = claimed_ids(&caller, &info, claims, settings);
    let provisioned = provisioning.filter(|_| caller.agent.is_none());
    let team_id_upsert = provisioning.is_some_and(|provisioning| provisioning.team_id_upsert);
    let team_settings = TeamSettings::from(&settings.config);
    let eligibility = policy.team_eligibility(&target);

    let claimed = team::claimed_team(
        host,
        claims,
        request,
        settings,
        rbac_team_id,
        team_id_upsert,
        &eligibility,
    )
    .await?;
    let granting = agent::delegated_team(host, caller.agent.as_ref(), &claimed, settings).await?;
    policy.granting_team(granting.as_deref(), claimed.team_id.as_deref(), &target)?;
    let (selected_team_id, selected_team) = team::selected(host, granting, &claimed).await?;
    policy.team(selected_team_id.as_deref(), selected_team.as_ref(), &target)?;

    let org = org::read(host, claims, settings).await?;
    let upsert = provisioned.is_some_and(|provisioning| provisioning.user_id_upsert);
    let ResolvedUser {
        user,
        user_id,
        membership: selected_membership,
    } = user::resolve_user(
        host,
        user_id.as_deref(),
        info.user_email.as_deref(),
        upsert,
        selected_team_id.as_deref(),
    )
    .await?;
    if user.as_ref().is_some_and(|user| user.deactivated) {
        return Err(JwtDenial::UserDeactivated { user_id }.into());
    }
    let end_user_id = claims::end_user_id(claims, settings);
    let end_user = match end_user_id.as_deref().filter(|text| !text.is_empty()) {
        Some(end_user_id) => Some(host.jwt_end_user(end_user_id).await?),
        None => None,
    };
    let (org_id, org) = match org {
        Some(org) => (Some(org.organization_id), Some(org.row)),
        None => (claims::org_id(claims, settings), None),
    };
    let user = match user {
        Some(user)
            if provisioned.is_some()
                && settings.config.sync_user_role_and_teams
                && settings.has_db =>
        {
            Some(user::sync_user(host, claims, user, settings).await?)
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
                request.model,
                &eligibility,
                team_id_upsert,
                team_settings,
            )
            .await?;
            if let Some(found) = &found {
                policy.team(Some(&found.team.team_id), Some(&found.team), &target)?;
            }
            joined(found)
        }
        None => joined(
            team::single_team_fallback(host, user_teams, user_id.as_deref(), team_id_upsert)
                .await?,
        ),
        Some(selected) => (Some(selected), selected_team, selected_membership),
    };
    let grants = match (provisioned, &user, &team) {
        (Some(_), Some(user), Some(team)) if !team.members.contains(&user.user_id) => {
            grants.with(Grant::AddTeamMember {
                user_id: user.user_id.clone(),
                team_id: team.team_id.clone(),
            })
        }
        _ => grants,
    };
    Ok(Principal {
        caller,
        user_id,
        user_email: info.user_email,
        user,
        team_id,
        team,
        membership,
        header_team: claimed.header_team,
        org_id,
        org,
        end_user_id,
        end_user,
        grants,
    })
}

fn identity_of<R: Rows>(claims: Value, principal: Principal<R>) -> Authorized<R> {
    let Principal {
        caller,
        user_id,
        user_email,
        user,
        team_id,
        team,
        membership,
        header_team: _,
        org_id,
        org,
        end_user_id,
        end_user,
        grants,
    } = principal;
    let is_proxy_admin = caller.agent.is_none()
        && user.as_ref().is_some_and(|user| {
            user.user_role.as_deref() == Some(<&str>::from(LitellmRole::ProxyAdmin))
        });
    let identity = JwtIdentity {
        claims,
        is_proxy_admin,
        user_id,
        user_email: user
            .as_ref()
            .and_then(|user| user.user_email.clone().filter(|text| !text.is_empty()))
            .or(user_email),
        team_id,
        org_id,
        end_user_id,
        agent_id: caller.agent_id,
        managed_agent: caller.agent,
        rows: JwtRows {
            user: user.map(|user| user.row),
            team: team.map(|team| team.row),
            membership: membership.filter(|_| !is_proxy_admin),
            org,
            end_user,
        },
    };
    Authorized { identity, grants }
}

/// A team's id, the team and the membership loaded with it.
type JoinedTeam<R> = (
    Option<String>,
    Option<Team<<R as Rows>::Team>>,
    Option<<R as Rows>::Membership>,
);

/// The team a fallback joined and the membership it loaded with it.
fn joined<R: Rows>(found: Option<Joined<R>>) -> JoinedTeam<R> {
    match found {
        Some(joined) => (
            Some(joined.team.team_id.clone()),
            Some(joined.team),
            joined.membership,
        ),
        None => (None, None, None),
    }
}

/// The token the host finishes for an identity, or the identity itself for a proxy admin.
async fn finished<H: AuthHost>(
    host: &H,
    identity: JwtIdentity<H::Rows>,
) -> Result<JwtOutcome<H::Rows>, Error> {
    if identity.is_proxy_admin {
        return Ok(JwtOutcome::Admin(identity));
    }
    Ok(JwtOutcome::Finished(
        host.finish_jwt(FinishedJwt::Identity(&identity)).await?,
    ))
}

/// The mapping for a claim no key was mapped to yet, written together with the other grants.
/// A proxy admin needs no key, which is cached so the next request skips the lookup.
async fn register<H: AuthHost>(
    host: &H,
    authorized: Authorized<H::Rows>,
    pending: &Pending,
    settings: &JwtSettings,
) -> Result<JwtOutcome<H::Rows>, Error> {
    let Authorized { identity, grants } = authorized;
    if identity.is_proxy_admin {
        commit::apply(host, grants).await?;
        mapping::remember_proxy_admin(host, pending, settings).await?;
        return Ok(JwtOutcome::Admin(identity));
    }
    let registration = mapping::registration(host, pending, &identity, settings).await?;
    let grants = grants.with(Grant::CreateKeyMapping(registration.clone()));
    let registered = match commit::apply(host, grants).await?.key_mapping {
        Some(write) => mapping::registered(host, &registration, write, settings).await?,
        None => None,
    };
    let Some((mapped, minted)) = registered else {
        return finished(host, identity).await;
    };
    let jwt = MappedJwt {
        claims: identity.claims,
        end_user_id: identity.end_user_id,
        user_email: identity.user_email,
        origin: KeyOrigin::AutoRegistered {
            minted,
            org_id: identity.org_id,
        },
        user: identity.rows.user,
    };
    let falls_through_to_key_checks = (settings.config.auto_register_map_existing_key
        || pending.behavior == Behavior::LinkOnly)
        && settings.has_master_key;
    if falls_through_to_key_checks {
        return Ok(JwtOutcome::MappedKey(mapped.key, jwt));
    }
    Ok(JwtOutcome::Finished(
        host.finish_jwt(FinishedJwt::AutoRegisteredKey(&jwt, &mapped.key.row))
            .await?,
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

    use std::{collections::BTreeSet, sync::LazyLock};

    use super::*;
    use crate::{
        AgentIdentityFailure, BudgetExceeded, ErrorType, KeyHash, PublicDenial, RbacDenial,
        TeamDenial, VerifyDenial,
        authn::{
            jwt::{MappedKey, NO_MAPPING_SENTINEL, PROXY_ADMIN_SENTINEL},
            key::{KeyRecord, KeySettings, Keyed},
        },
        identity::jwt::{
            AliasRead, ManagedAgent, Org, RegisteredAgent, TeamRead, TeamSync, User, team_sync,
        },
        routes::passthrough::{ForwardedRoutes, PassthroughGrants},
        test_support::route_groups,
        testing::{Labels, ScriptedHost},
    };

    static KEY: LazyLock<KeySettings> = LazyLock::new(KeySettings::default);

    fn policy(settings: &JwtSettings) -> RoutePolicy<'_> {
        RoutePolicy::new(&KEY, settings)
    }

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
        super::authorize_jwt(
            host,
            &Token::new(TOKEN),
            &request,
            settings,
            &policy(settings),
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
    async fn email_domain_is_checked_before_the_org() {
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
        assert_eq!(calls(&host), ["managed_agent"]);
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
                "delegated a1 team=- explicit=false allowed=-",
                "granting_team t5",
                "resolve_user u1 team=t5",
                "record_agent agent a1",
            ]
        );
    }

    /// A managed agent's authentication is recorded only once the request is allowed. An agent a
    /// later check refuses is never recorded, and that check's denial wins over a failed recording.
    #[rstest]
    #[case::recording_works(None)]
    #[case::recording_would_fail(Some(AgentIdentityFailure { policy_unavailable: true, message: "down".to_owned() }))]
    #[tokio::test]
    async fn a_refused_managed_agent_is_never_recorded(
        #[case] failure: Option<AgentIdentityFailure>,
    ) {
        let host = ScriptedHost {
            managed: Some(agent(AgentMode::Delegated)),
            delegated_team: Some("t5"),
            agent_failure: failure,
            teams: HashMap::from([("t5", team("t5"))]),
            ..host(json!({"iss": "https://idp", "sub": "someone-else"}))
        };
        static GET: http::Method = http::Method::GET;
        let discovery = JwtRequest {
            method: Some(&GET),
            ..request("/v1/agents")
        };
        let denial = authorize_with(&host, discovery, &settings(config()), PROVISIONED).await;
        assert_eq!(denial, Err(JwtDenial::GrantingTeamRouteNotAllowed.into()));
        assert_eq!(
            calls(&host),
            [
                "managed_agent",
                "delegated a1 team=- explicit=false allowed=-"
            ]
        );
    }

    /// A rejected email domain is refused before any team is read or upserted. It used to run
    /// after the team reads, the pass-through gate and the organization read.
    #[tokio::test]
    async fn a_rejected_email_domain_reads_no_team() {
        let host = ScriptedHost {
            teams: HashMap::from([("t1", team("t1"))]),
            ..host(json!({"sub": "u1", "email": "u1@evil.com", "groups": ["t1"]}))
        };
        let settings = settings(JwtAuthSettings {
            team_ids_jwt_field: Some("groups".into()),
            user_allowed_email_domain: Some("corp.com".to_owned()),
            ..config()
        });
        let provisioning = Some(Provisioning {
            user_id_upsert: true,
            team_id_upsert: true,
        });
        let denial = authorize_with(&host, request(CHAT), &settings, provisioning).await;
        assert!(matches!(
            denial,
            Err(Error::Jwt(JwtDenial::EmailDomainNotAllowed { .. }))
        ));
        assert_eq!(calls(&host), ["managed_agent"]);
    }

    /// `auth_builder` never applied the email domain rule to a proxy admin, and still does not.
    #[tokio::test]
    async fn a_proxy_admin_skips_the_email_domain_rule() {
        let host = ScriptedHost {
            users: HashMap::from([("u1", user("u1", "internal_user", &[]))]),
            ..host(json!({"sub": "u1", "email": "u1@evil.com", "scope": "litellm_proxy_admin"}))
        };
        let settings = settings(JwtAuthSettings {
            admin_allowed_routes: Box::new(["openai_routes".to_owned()]),
            user_allowed_email_domain: Some("corp.com".to_owned()),
            ..config()
        });
        let identity = authorize_with(&host, request(CHAT), &settings, PROVISIONED).await;
        assert!(identity.is_ok_and(|identity| identity.is_proxy_admin));
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
        let settings = settings(config());
        let identity = super::resolve_identity(
            &host,
            &Token::new(TOKEN),
            Timestamp::UNIX_EPOCH,
            &settings,
            &policy(&settings),
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
        build(
            host,
            &Token::new(TOKEN),
            &request(route),
            settings,
            &policy(settings),
        )
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

    /// A deactivated user is refused right after it is read, before its role and teams are synced
    /// or it is added to a team. It used to be refused after those writes.
    #[tokio::test]
    async fn a_deactivated_user_gets_no_membership_writes() {
        let host = ScriptedHost {
            teams: HashMap::from([("t2", team("t2"))]),
            users: HashMap::from([(
                "u1",
                User {
                    deactivated: true,
                    ..user("u1", "internal_user", &["t1"])
                },
            )]),
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
        let denial = authorize_with(&host, request(CHAT), &settings, PROVISIONED).await;
        assert_eq!(
            denial,
            Err(JwtDenial::UserDeactivated {
                user_id: Some("u1".to_owned()),
            }
            .into())
        );
        assert_eq!(
            calls(&host),
            ["managed_agent", "team t2", "resolve_user u1 team=t2"]
        );
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
        let settings = settings(config());
        let denial = super::authorize_jwt(
            &host,
            &Token::new("a.b.c"),
            &request(CHAT),
            &settings,
            &policy(&settings),
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
