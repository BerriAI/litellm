use litellm_config::LitellmRole;

use crate::{
    AuthHost, Error, RouteDenial,
    passthrough::{allowed_routes_grant, first_denied_route, jwt_team_routes_grant, matches_prefix, matches_wildcard},
    routes::{group_lists, is_route_allowed, matches_any_template},
};

const TEAM_ORG_CONTEXT_ROUTE: &str = "/team/update";
const PATCH_TEAM_ROUTE_TEMPLATE: &str = "/team/{team_id}";

/// Management writes a proxy admin viewer may never call, even with a method that reads.
const VIEWER_BLOCKED_MANAGEMENT_ROUTES: &[&str] = &[
    "/user/new",
    "/management/v1/users/bulk",
    "/user/delete",
    "/management/v1/users/bulk_delete",
    "/user/bulk_update",
    "/team/new",
    "/management/v1/teams/{team_id}/members/bulk_delete",
    "/management/v1/teams/{team_id}/members/bulk_update",
    "/team/update",
    "/team/delete",
    "/team/block",
    "/team/unblock",
    "/team/permissions_update",
    "/team/permissions_bulk_update",
    "/model/new",
    "/model/update",
    "/model/delete",
    "/jwt/key/mapping/new",
    "/jwt/key/mapping/update",
    "/jwt/key/mapping/delete",
    "/key/generate",
    "/key/update",
    "/key/delete",
    "/key/regenerate",
    "/key/service-account/generate",
    "/key/block",
    "/key/unblock",
    "/key/bulk_update",
    "/team/key/bulk_update",
];
const VIEWER_BLOCKED_KEY_SUFFIXES: &[&str] = &["/regenerate", "/reset_spend"];
/// Writes a proxy admin viewer may never call, whatever the method.
const VIEWER_BLOCKED_WRITE_ROUTES: &[&str] = &[
    "/user/new",
    "/management/v1/users/bulk",
    "/user/delete",
    "/management/v1/users/bulk_delete",
    "/user/bulk_update",
    "/team/new",
    "/management/v1/teams/{team_id}/members/bulk_delete",
    "/management/v1/teams/{team_id}/members/bulk_update",
    "/team/update",
    "/team/delete",
    "/model/new",
    "/model/update",
    "/model/delete",
    "/key/generate",
    "/key/delete",
    "/key/update",
    "/key/regenerate",
    "/key/service-account/generate",
    "/key/block",
    "/key/unblock",
    "/team/key/bulk_update",
];
/// Writes that only touch the caller's own credential row, by method.
const SELF_SCOPED_CREDENTIAL_ROUTES: &[(&str, &str)] = &[
    ("GET", "/credentials/user_connections"),
    ("POST", "/credentials/{credential_name:path}/user_connection/start"),
    ("POST", "/credentials/{credential_name:path}/user_connection/poll"),
    ("DELETE", "/credentials/{credential_name:path}/user_connection"),
    ("POST", "/v1/mcp/server/oauth/{server_id}/register"),
    ("POST", "/v1/mcp/server/{server_id}/oauth-user-credential"),
    ("DELETE", "/v1/mcp/server/{server_id}/oauth-user-credential"),
    ("POST", "/v1/mcp/server/{server_id}/user-credential"),
    ("DELETE", "/v1/mcp/server/{server_id}/user-credential"),
    ("POST", "/v1/mcp/server/{server_id}/user-env-vars"),
    ("DELETE", "/v1/mcp/server/{server_id}/user-env-vars"),
];

#[derive(Clone, Copy, Debug)]
pub struct OrgMembership<'a> {
    pub organization_id: &'a str,
    pub role: Option<&'a str>,
}

/// What the route checks read from the user row.
#[derive(Clone, Copy, Debug, Default)]
pub struct RouteUser<'a> {
    pub user_id: Option<&'a str>,
    pub user_role: Option<&'a str>,
    /// `organization_memberships`, unset when the row has none.
    pub organization_memberships: Option<&'a [OrgMembership<'a>]>,
}

/// What the route checks read from the authenticated token.
#[derive(Clone, Copy, Debug, Default)]
pub struct RouteToken<'a> {
    pub user_id: Option<&'a str>,
    /// `"get_spend_routes" in permissions`
    pub can_read_spend: bool,
    pub is_team_service_account: bool,
    pub allowed_routes: Option<&'a [String]>,
    /// The key's `allowed_passthrough_routes`.
    pub key_passthrough_routes: Option<&'a [String]>,
    /// The team's `allowed_passthrough_routes`.
    pub team_passthrough_routes: Option<&'a [String]>,
    /// The key's then the team's `denied_passthrough_routes`.
    pub denied_passthrough_routes: &'a [String],
    /// `litellm_jwtauth.team_allowed_routes`, for a team token JWT auth built.
    pub jwt_team_allowed_routes: Option<&'a [String]>,
}

/// What the route checks read from the request.
#[derive(Clone, Copy, Debug, Default)]
pub struct RouteRequest<'a> {
    pub route: &'a str,
    /// Upper-cased.
    pub method: Option<&'a str>,
    pub route_template: Option<&'a str>,
    /// The `user_id` query parameter.
    pub query_user_id: Option<&'a str>,
    /// The body's keys, in order.
    pub body_keys: &'a [String],
    pub organization_id: Option<&'a str>,
    pub organizations: Option<&'a [String]>,
    pub team_id: Option<&'a str>,
    pub is_llm_api_route: bool,
    /// `RouteChecks.is_auth_enforced_pass_through_route`
    pub auth_enforced_pass_through: bool,
    pub forwarded_routes: &'a [String],
    /// `general_settings.admin_only_routes`, on a premium proxy.
    pub admin_only_routes: &'a [String],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RouteAccess<'a> {
    pub request: RouteRequest<'a>,
    pub user: Option<RouteUser<'a>>,
    /// Unset when the caller authorized without a token.
    pub token: Option<RouteToken<'a>>,
}

fn shown(value: Option<&str>) -> &str {
    value.unwrap_or("unset")
}

/// `organization_role_based_access_check`
pub fn organization_access(access: &RouteAccess<'_>) -> Result<(), RouteDenial> {
    let Some(user) = access.user else {
        return Ok(());
    };
    let route = access.request.route;
    let organization_id = access.request.organization_id;
    let is_proxy_admin = user.user_role == Some(<&str>::from(LitellmRole::ProxyAdmin));
    if route == "/organization/new" && !is_proxy_admin {
        return Err(RouteDenial::OrganizationNewNeedsAdmin {
            role: shown(user.user_role).to_owned(),
        });
    }
    if is_proxy_admin {
        return Ok(());
    }
    let memberships = user.organization_memberships;
    let role_in = |organization_id: &str| {
        memberships
            .unwrap_or_default()
            .iter()
            .rev()
            .find(|membership| membership.organization_id == organization_id)
            .and_then(|membership| membership.role)
    };
    let org_admin = <&str>::from(LitellmRole::OrgAdmin);
    if group_lists(route, "org_admin_only_routes") {
        if memberships.is_none() {
            return Err(RouteDenial::NoOrganization {
                route: route.to_owned(),
            });
        }
        let Some(organization_id) = organization_id else {
            return Err(RouteDenial::OrganizationIdMissing);
        };
        let Some(role) = role_in(organization_id) else {
            return Err(RouteDenial::NoOrganizationRole {
                organization_id: organization_id.to_owned(),
            });
        };
        if role != org_admin {
            return Err(RouteDenial::NotOrganizationAdmin {
                route: route.to_owned(),
                organization_id: organization_id.to_owned(),
                role: role.to_owned(),
            });
        }
        return Ok(());
    }
    if route != "/team/new" || memberships.is_none_or(<[OrgMembership<'_>]>::is_empty) {
        return Ok(());
    }
    let Some(organization_id) = organization_id else {
        return Err(RouteDenial::TeamOrganizationMissing {
            organizations: memberships
                .unwrap_or_default()
                .iter()
                .map(|membership| membership.organization_id.to_owned())
                .collect(),
        });
    };
    let role = role_in(organization_id);
    if role != Some(org_admin) {
        return Err(RouteDenial::NotTeamOrganizationAdmin {
            route: route.to_owned(),
            organization_id: organization_id.to_owned(),
            role: shown(role).to_owned(),
        });
    }
    Ok(())
}

/// `user_is_org_admin`: an org admin of every organization the request names.
pub fn is_org_admin(
    user: Option<RouteUser<'_>>,
    organization_id: Option<&str>,
    organizations: Option<&[String]>,
) -> bool {
    let Some(memberships) = user.and_then(|user| user.organization_memberships) else {
        return false;
    };
    let mut candidates = organization_id
        .into_iter()
        .chain(organizations.unwrap_or_default().iter().map(String::as_str))
        .peekable();
    candidates.peek().is_some()
        && candidates.all(|candidate| {
            memberships.iter().any(|membership| {
                membership.organization_id == candidate
                    && membership.role == Some(<&str>::from(LitellmRole::OrgAdmin))
            })
        })
}

fn is_self_scoped_credential_route(route: &str, method: Option<&str>) -> bool {
    method.is_some_and(|method| {
        SELF_SCOPED_CREDENTIAL_ROUTES
            .iter()
            .any(|(expected, template)| *expected == method && matches_any_template(route, &[template]))
    })
}

fn effective_role(user: Option<RouteUser<'_>>) -> Option<LitellmRole> {
    user.map(|user| {
        user.user_role
            .and_then(|role| role.parse().ok())
            .unwrap_or(LitellmRole::InternalUser)
    })
}

fn only_user_email(request: &RouteRequest<'_>, role: &'static str) -> Result<(), RouteDenial> {
    match request.body_keys.iter().find(|key| *key != "user_email") {
        Some(param) => Err(RouteDenial::ViewerInvalidParam {
            role,
            route: request.route.to_owned(),
            param: param.clone(),
        }),
        None => Ok(()),
    }
}

/// `_check_proxy_admin_viewer_access`: everything a proxy admin reads, nothing that writes or costs.
pub fn viewer_access(request: &RouteRequest<'_>) -> Result<(), RouteDenial> {
    let role = <&str>::from(LitellmRole::ProxyAdminViewOnly);
    let route = request.route;
    let denied = || RouteDenial::ViewerNotAllowed {
        role,
        route: route.to_owned(),
    };
    if request.is_llm_api_route {
        return Err(RouteDenial::ViewerInference { role });
    }
    if is_route_allowed(route, &["management_routes"]) {
        if route == "/user/update" {
            only_user_email(request, role)?;
        } else if matches_any_template(route, VIEWER_BLOCKED_MANAGEMENT_ROUTES)
            || (route.starts_with("/key/") && VIEWER_BLOCKED_KEY_SUFFIXES.iter().any(|suffix| route.ends_with(suffix)))
        {
            return Err(denied());
        }
    }
    if matches!(request.method.unwrap_or("GET"), "GET" | "HEAD" | "OPTIONS") {
        return Ok(());
    }
    if route == "/user/update" {
        return only_user_email(request, role);
    }
    if matches!(route, "/user/password/change" | "/session/logout")
        || is_self_scoped_credential_route(route, request.method)
    {
        return Ok(());
    }
    if matches_any_template(route, VIEWER_BLOCKED_WRITE_ROUTES)
        || (route.starts_with("/key/") && route.ends_with("/regenerate"))
    {
        return Err(denied());
    }
    if is_route_allowed(route, &["admin_viewer_routes", "global_spend_tracking_routes"]) {
        return Ok(());
    }
    Err(denied())
}

/// `RouteChecks._mask_user_id`
pub fn masked_user_id(user_id: &str) -> String {
    let chars: Vec<char> = user_id.chars().collect();
    match chars.len() {
        0..=4 => "***".to_owned(),
        length if length <= 8 => "*".repeat(length),
        length => format!(
            "{}{}{}",
            chars[..6].iter().collect::<String>(),
            "*".repeat(length - 8),
            chars[length - 2..].iter().collect::<String>()
        ),
    }
}

fn admin_only(route: &str, user: Option<RouteUser<'_>>) -> RouteDenial {
    RouteDenial::AdminOnly {
        route: route.to_owned(),
        role: user
            .and_then(|user| user.user_role)
            .filter(|role| !role.is_empty())
            .unwrap_or("unknown")
            .to_owned(),
        user_id: masked_user_id(
            user.and_then(|user| user.user_id)
                .filter(|user_id| !user_id.is_empty())
                .unwrap_or("unknown"),
        ),
    }
}

/// `_require_auth_pass_through_access`: a deny list match refuses even a granted route.
fn auth_pass_through_access(route: &str, token: &RouteToken<'_>, forwarded: &[String]) -> Result<(), RouteDenial> {
    if !token.denied_passthrough_routes.is_empty()
        && let Some(denied_route) = first_denied_route(forwarded, token.denied_passthrough_routes)
    {
        return Err(RouteDenial::PassThroughDenied {
            route: route.to_owned(),
            denied_route: denied_route.to_owned(),
        });
    }
    if allowed_routes_grant(route, token.key_passthrough_routes, token.team_passthrough_routes)
        || token
            .jwt_team_allowed_routes
            .is_some_and(|routes| jwt_team_routes_grant(route, routes))
    {
        return Ok(());
    }
    Err(RouteDenial::PassThroughNotAllowed {
        route: route.to_owned(),
    })
}

/// `RouteChecks.non_proxy_admin_allowed_routes_check`
pub fn non_admin_access(
    access: &RouteAccess<'_>,
    token: &RouteToken<'_>,
    organization_id: Option<&str>,
) -> Result<(), RouteDenial> {
    let request = &access.request;
    let route = request.route;
    let role = effective_role(access.user);
    if request.admin_only_routes.iter().any(|admin_route| admin_route == route) {
        return Err(RouteDenial::CustomAdminOnly {
            route: route.to_owned(),
        });
    }
    if request.auth_enforced_pass_through {
        return auth_pass_through_access(route, token, request.forwarded_routes);
    }
    if request.is_llm_api_route {
        return Ok(());
    }
    if group_lists(route, "info_routes") {
        if route == "/user/info"
            && let Some(user_id) = request.query_user_id.filter(|user_id| !user_id.is_empty())
            && Some(user_id) != token.user_id
            && role != Some(LitellmRole::ProxyAdminViewOnly)
        {
            return Err(RouteDenial::OtherUsersInfo {
                user_id: user_id.to_owned(),
                key_user_id: shown(token.user_id).to_owned(),
            });
        }
        return Ok(());
    }
    if group_lists(route, "global_spend_tracking_routes") && token.can_read_spend {
        return Ok(());
    }
    if route == "/lens" || route.starts_with("/lens/") {
        return Ok(());
    }
    if role == Some(LitellmRole::ProxyAdminViewOnly) {
        return viewer_access(request);
    }
    let role_grants = (role == Some(LitellmRole::InternalUser)
        && (is_route_allowed(route, &["internal_user_routes"]) || is_self_scoped_credential_route(route, request.method)))
        || (is_org_admin(access.user, organization_id, request.organizations)
            && is_route_allowed(route, &["org_admin_allowed_routes"]))
        || (role == Some(LitellmRole::InternalUserViewOnly)
            && is_route_allowed(route, &["internal_user_view_only_routes"]))
        || is_route_allowed(route, &["self_managed_routes"]);
    if role_grants || route.starts_with("/v1/mcp/") || route.starts_with("/mcp-rest/") {
        return Ok(());
    }
    if allowed_routes_grant(route, token.key_passthrough_routes, token.team_passthrough_routes)
        || (token.is_team_service_account && is_route_allowed(route, &["team_service_account_key_routes"]))
    {
        return Ok(());
    }
    match token.allowed_routes {
        Some(allowed) if allowed.iter().any(|allowed| matches_prefix(route, allowed) || matches_wildcard(route, allowed)) => Ok(()),
        _ => Err(admin_only(route, access.user)),
    }
}

/// The team a `/team/update` body or a `/team/{team_id}` path names, when the request names no
/// organization itself.
pub fn org_context_team<'a>(request: &RouteRequest<'a>) -> Option<&'a str> {
    if request.organization_id.is_some_and(|id| !id.is_empty()) {
        return None;
    }
    let team_id = match request.route {
        TEAM_ORG_CONTEXT_ROUTE => request.team_id,
        route if request.route_template == Some(PATCH_TEAM_ROUTE_TEMPLATE) => route.rsplit('/').next(),
        _ => None,
    };
    team_id.filter(|team_id| !team_id.is_empty())
}

/// `organization_role_based_access_check`, then whether the identity may call the route, with the
/// organization of the team the request names standing in for one the request does not pass.
pub async fn route_access<H: AuthHost>(host: &H, access: &RouteAccess<'_>) -> Result<(), Error> {
    organization_access(access)?;
    let team_org = match org_context_team(&access.request) {
        Some(team_id) => host.team_organization(team_id).await?.filter(|id| !id.is_empty()),
        None => None,
    };
    Ok(route_allowed(access, team_org.as_deref().or(access.request.organization_id))?)
}

/// `_is_api_route_allowed`: a proxy admin calls any route, anyone else only the routes
/// `non_admin_access` lets through. `organization_id` is the organization the request acts in.
pub fn route_allowed(access: &RouteAccess<'_>, organization_id: Option<&str>) -> Result<(), RouteDenial> {
    let Some(token) = access.token else {
        return Err(RouteDenial::NoToken);
    };
    if access
        .user
        .is_some_and(|user| user.user_role == Some(<&str>::from(LitellmRole::ProxyAdmin)))
    {
        return Ok(());
    }
    non_admin_access(access, &token, organization_id)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::{PublicDenial, testing::ScriptedHost};

    fn owned(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn user(role: &'static str) -> Option<RouteUser<'static>> {
        Some(RouteUser {
            user_id: Some("user-123456789"),
            user_role: Some(role),
            organization_memberships: None,
        })
    }

    async fn check(access: &RouteAccess<'_>) -> Option<PublicDenial> {
        route_access(&ScriptedHost::default(), access).await.err().and_then(|error| error.public())
    }

    fn access<'a>(route: &'a str, method: &'a str, user: Option<RouteUser<'a>>) -> RouteAccess<'a> {
        RouteAccess {
            request: RouteRequest {
                route,
                method: Some(method),
                ..RouteRequest::default()
            },
            user,
            token: Some(RouteToken::default()),
        }
    }

    #[rstest]
    #[case::short("abcd", "***")]
    #[case::up_to_eight("abcdefgh", "********")]
    #[case::long("user-123456789", "user-1******89")]
    fn user_ids_are_masked_like_the_proxy(#[case] user_id: &str, #[case] masked: &str) {
        assert_eq!(masked_user_id(user_id), masked);
    }

    #[rstest]
    #[case::internal_user_own_route("/key/generate", "POST", "internal_user", true)]
    #[case::internal_user_admin_route("/user/new", "POST", "internal_user", false)]
    #[case::viewer_reads("/team/list", "GET", "proxy_admin_viewer", true)]
    #[case::viewer_writes("/team/block", "POST", "proxy_admin_viewer", false)]
    #[case::proxy_admin_anything("/user/new", "POST", "proxy_admin", true)]
    #[case::lens("/lens/datasets", "POST", "internal_user", true)]
    #[case::mcp_rest("/mcp-rest/tools/list", "GET", "internal_user_viewer", true)]
    #[tokio::test]
    async fn roles_reach_their_routes_only(
        #[case] route: &str,
        #[case] method: &str,
        #[case] role: &'static str,
        #[case] allowed: bool,
    ) {
        assert_eq!(check(&access(route, method, user(role))).await.is_none(), allowed);
    }

    #[tokio::test]
    async fn the_admin_only_denial_names_the_role_and_a_masked_user() {
        assert_eq!(
            check(&access("/user/new", "POST", user("internal_user"))).await,
            Some(PublicDenial::Proxy {
                status: 401,
                message: "Authentication Error, Only proxy admin can be used to generate, delete, update info for new keys/users/teams. Route=/user/new. Your role=internal_user. Your user_id=user-1******89".to_owned(),
                error_type: crate::ErrorType::Auth,
                param: None,
            })
        );
    }

    #[rstest]
    #[case::own_info(Some("me"), true)]
    #[case::someone_elses(Some("other"), false)]
    #[case::no_param(None, true)]
    #[tokio::test]
    async fn user_info_only_shows_the_keys_own_user(#[case] query_user_id: Option<&str>, #[case] allowed: bool) {
        let access = RouteAccess {
            request: RouteRequest {
                route: "/user/info",
                method: Some("GET"),
                query_user_id,
                ..RouteRequest::default()
            },
            user: user("internal_user"),
            token: Some(RouteToken {
                user_id: Some("me"),
                ..RouteToken::default()
            }),
        };
        assert_eq!(check(&access).await.is_none(), allowed);
    }

    #[rstest]
    #[case::granted(&["/svc"], &[], true)]
    #[case::denied_even_when_granted(&["/svc"], &["/svc"], false)]
    #[case::not_granted(&[], &[], false)]
    #[tokio::test]
    async fn auth_enforced_pass_through_needs_an_explicit_grant(
        #[case] allowed: &[&str],
        #[case] denied: &[&str],
        #[case] reachable: bool,
    ) {
        let (allowed, denied, forwarded) = (owned(allowed), owned(denied), owned(&["/svc/a"]));
        let access = RouteAccess {
            request: RouteRequest {
                route: "/svc/a",
                method: Some("POST"),
                auth_enforced_pass_through: true,
                forwarded_routes: &forwarded,
                ..RouteRequest::default()
            },
            user: user("internal_user"),
            token: Some(RouteToken {
                key_passthrough_routes: Some(&allowed),
                denied_passthrough_routes: &denied,
                ..RouteToken::default()
            }),
        };
        assert_eq!(check(&access).await.is_none(), reachable);
    }

    #[rstest]
    #[case::org_admin_of_the_named_org(Some("o1"), "org_admin", None)]
    #[case::member_but_not_admin(Some("o1"), "internal_user", Some(401))]
    #[case::no_org_passed(None, "org_admin", Some(401))]
    #[tokio::test]
    async fn org_admin_only_routes_need_the_admin_role_in_the_named_org(
        #[case] organization_id: Option<&str>,
        #[case] org_role: &str,
        #[case] status: Option<u16>,
    ) {
        let memberships = [OrgMembership {
            organization_id: "o1",
            role: Some(org_role),
        }];
        let access = RouteAccess {
            request: RouteRequest {
                route: "/organization/member_add",
                method: Some("POST"),
                organization_id,
                ..RouteRequest::default()
            },
            user: Some(RouteUser {
                user_id: Some("u"),
                user_role: Some("internal_user"),
                organization_memberships: Some(&memberships),
            }),
            token: Some(RouteToken::default()),
        };
        let denial = organization_access(&access).err().map(|denial| match denial.public() {
            PublicDenial::Proxy { status, .. } => status,
            other => panic!("{other:?}"),
        });
        assert_eq!(denial, status);
    }

    #[tokio::test]
    async fn a_team_update_borrows_the_teams_organization_for_an_org_admin() {
        let memberships = [OrgMembership {
            organization_id: "org-of-team",
            role: Some("org_admin"),
        }];
        let host = ScriptedHost {
            team_organizations: std::collections::HashMap::from([("t1", "org-of-team")]),
            ..ScriptedHost::default()
        };
        let access = RouteAccess {
            request: RouteRequest {
                route: "/team/update",
                method: Some("POST"),
                team_id: Some("t1"),
                ..RouteRequest::default()
            },
            user: Some(RouteUser {
                user_id: Some("u"),
                user_role: Some("internal_user"),
                organization_memberships: Some(&memberships),
            }),
            token: Some(RouteToken::default()),
        };
        assert_eq!(route_access(&host, &access).await, Ok(()));
        assert_eq!(host.calls(), ["team_organization t1"]);
    }
}
