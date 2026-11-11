use super::{JwtSettings, RbacDenial};
use crate::routes::{actual_routes, is_route_allowed};
use litellm_config::RbacRole;

fn role_permissions(
    role: RbacRole,
    settings: &JwtSettings,
) -> Option<&litellm_config::RolePermissions> {
    settings
        .role_permissions
        .iter()
        .find(|permissions| permissions.role == role)
}

/// `get_role_based_models`: the first `role_permissions` entry for the role decides.
pub fn role_based_models(role: RbacRole, settings: &JwtSettings) -> Option<&[String]> {
    role_permissions(role, settings)?.models.as_deref()
}

/// `get_role_based_routes`
pub fn role_based_routes(role: RbacRole, settings: &JwtSettings) -> Option<&[String]> {
    role_permissions(role, settings)?.routes.as_deref()
}

/// `JWTAuthManager.can_rbac_role_call_route`
pub fn can_role_call_route(
    role: RbacRole,
    route: &str,
    settings: &JwtSettings,
) -> Result<(), RbacDenial> {
    match role_based_routes(role, settings) {
        Some(allowed) if !is_route_allowed(route, allowed) => Err(RbacDenial::RoleRoute {
            role,
            route: route.to_owned(),
            allowed: allowed.into(),
        }),
        _ => Ok(()),
    }
}

/// `JWTAuthManager.can_rbac_role_call_model`
pub fn can_role_call_model(
    role: RbacRole,
    model: Option<&str>,
    settings: &JwtSettings,
) -> Result<(), RbacDenial> {
    match (role_based_models(role, settings), model) {
        (Some(allowed), Some(model)) if !allowed.iter().any(|allowed| allowed == model) => {
            Err(RbacDenial::RoleModel {
                role,
                model: model.to_owned(),
                allowed: allowed.into(),
            })
        }
        _ => Ok(()),
    }
}

/// `JWTAuthManager.check_scope_based_access`: the models of every `scope_mappings` entry whose
/// scope the token has. The caller gates it on `enforce_scope_based_access`.
pub fn check_scope_based_access(
    scopes: &[String],
    model: Option<&str>,
    settings: &JwtSettings,
) -> Result<(), RbacDenial> {
    let mappings = settings
        .config
        .scope_mappings
        .as_deref()
        .unwrap_or_default();
    let Some(model) = model.filter(|model| !mappings.is_empty() && !model.is_empty()) else {
        return Ok(());
    };
    let allowed: Vec<String> = mappings
        .iter()
        .filter(|mapping| scopes.contains(&mapping.scope))
        .flat_map(|mapping| {
            mapping
                .models
                .as_deref()
                .unwrap_or_default()
                .iter()
                .cloned()
        })
        .collect();
    if allowed.iter().any(|allowed| allowed == model) {
        return Ok(());
    }
    Err(RbacDenial::ScopeModel {
        model: model.to_owned(),
        allowed,
    })
}

/// `JWTAuthManager.check_rbac_role`: with `enforce_rbac`, the role's models, then its routes.
pub fn check_rbac_role(
    role: Option<RbacRole>,
    model: Option<&str>,
    route: &str,
    settings: &JwtSettings,
) -> Result<(), RbacDenial> {
    if !settings.config.enforce_rbac {
        return Ok(());
    }
    let role = role.ok_or(RbacDenial::UnmatchedRole)?;
    can_role_call_model(role, model, settings)?;
    can_role_call_route(role, route, settings)
}

/// `allowed_routes_check`: admins against `admin_allowed_routes`, teams against
/// `team_allowed_routes`, anyone else never.
pub fn allowed_routes_check(role: RbacRole, route: &str, settings: &JwtSettings) -> bool {
    match role {
        RbacRole::ProxyAdmin => is_route_allowed(route, &settings.config.admin_allowed_routes),
        RbacRole::Team => is_route_allowed(route, &settings.config.team_allowed_routes),
        RbacRole::InternalUser => false,
    }
}

/// The route half of `JWTAuthManager.check_admin_access`, for a token that has the admin scope.
pub fn check_admin_route(route: &str, settings: &JwtSettings) -> Result<(), RbacDenial> {
    if allowed_routes_check(RbacRole::ProxyAdmin, route, settings) {
        return Ok(());
    }
    Err(RbacDenial::AdminRoute {
        route: route.to_owned(),
        allowed: actual_routes(&settings.config.admin_allowed_routes),
    })
}

/// `JWTAuthManager._is_team_route_allowed`. `auth_enforced_pass_through` is
/// `RouteChecks.is_auth_enforced_pass_through_route(route, method)`, which reads the host's
/// registered pass-through endpoints.
pub fn is_team_route_allowed(
    route: &str,
    auth_enforced_pass_through: bool,
    settings: &JwtSettings,
) -> bool {
    auth_enforced_pass_through || allowed_routes_check(RbacRole::Team, route, settings)
}

#[cfg(test)]
mod tests {
    use litellm_config::{JwtAuthSettings, RolePermissions, ScopeMapping};
    use rstest::rstest;

    use super::*;

    fn strings(items: &[&str]) -> Box<[String]> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    fn settings() -> JwtSettings {
        JwtSettings {
            config: JwtAuthSettings {
                enforce_rbac: true,
                scope_mappings: Some(Box::new([
                    ScopeMapping {
                        scope: "read".to_owned(),
                        models: Some(strings(&["gpt-a"])),
                        routes: None,
                    },
                    ScopeMapping {
                        scope: "write".to_owned(),
                        models: Some(strings(&["gpt-b"])),
                        routes: None,
                    },
                ])),
                ..JwtAuthSettings::default()
            },
            role_permissions: Box::new([
                RolePermissions {
                    role: RbacRole::Team,
                    models: Some(strings(&["gpt-a"])),
                    routes: Some(strings(&["openai_routes", "/custom/*"])),
                },
                RolePermissions {
                    role: RbacRole::Team,
                    models: None,
                    routes: None,
                },
                RolePermissions {
                    role: RbacRole::InternalUser,
                    models: None,
                    routes: None,
                },
            ]),
            ..JwtSettings::default()
        }
    }

    #[rstest]
    #[case::allowed(RbacRole::Team, Some("gpt-a"), "/chat/completions", Ok(()))]
    #[case::group_template(RbacRole::Team, Some("gpt-a"), "/openai/deployments/my-model/chat/completions", Ok(()))]
    #[case::wildcard(RbacRole::Team, None, "/custom/x/y", Ok(()))]
    #[case::unlisted_role_is_unrestricted(RbacRole::ProxyAdmin, Some("anything"), "/key/generate", Ok(()))]
    #[case::role_without_lists(RbacRole::InternalUser, Some("anything"), "/key/generate", Ok(()))]
    #[case::model_checked_before_route(
        RbacRole::Team,
        Some("gpt-x"),
        "/key/generate",
        Err(RbacDenial::RoleModel { role: RbacRole::Team, model: "gpt-x".to_owned(), allowed: strings(&["gpt-a"]) })
    )]
    #[case::route(
        RbacRole::Team,
        None,
        "/key/generate",
        Err(RbacDenial::RoleRoute {
            role: RbacRole::Team,
            route: "/key/generate".to_owned(),
            allowed: strings(&["openai_routes", "/custom/*"]),
        })
    )]
    fn rbac_role_checks_models_then_routes(
        #[case] role: RbacRole,
        #[case] model: Option<&str>,
        #[case] route: &str,
        #[case] expected: Result<(), RbacDenial>,
    ) {
        assert_eq!(
            check_rbac_role(Some(role), model, route, &settings()),
            expected
        );
    }

    #[test]
    fn rbac_requires_a_role_when_enforced() {
        assert_eq!(
            check_rbac_role(None, None, "/chat/completions", &settings()),
            Err(RbacDenial::UnmatchedRole)
        );
    }

    #[test]
    fn rbac_off_allows_anything() {
        let settings = JwtSettings {
            config: JwtAuthSettings::default(),
            ..settings()
        };
        assert_eq!(
            check_rbac_role(None, Some("gpt-x"), "/key/generate", &settings),
            Ok(())
        );
    }

    #[rstest]
    #[case::no_model(&[], None, Ok(()))]
    #[case::empty_model(&[], Some(""), Ok(()))]
    #[case::scope_grants(&["write"], Some("gpt-b"), Ok(()))]
    #[case::other_scope(
        &["read", "other"],
        Some("gpt-b"),
        Err(RbacDenial::ScopeModel { model: "gpt-b".to_owned(), allowed: vec!["gpt-a".to_owned()] })
    )]
    fn scope_based_access(
        #[case] scopes: &[&str],
        #[case] model: Option<&str>,
        #[case] expected: Result<(), RbacDenial>,
    ) {
        let scopes: Vec<String> = scopes.iter().map(|scope| (*scope).to_owned()).collect();
        assert_eq!(
            check_scope_based_access(&scopes, model, &settings()),
            expected
        );
    }

    #[test]
    fn scope_based_access_without_mappings_allows() {
        let settings = JwtSettings {
            config: JwtAuthSettings::default(),
            ..settings()
        };
        assert_eq!(
            check_scope_based_access(&[], Some("gpt-x"), &settings),
            Ok(())
        );
    }

    #[rstest]
    #[case::admin_management(RbacRole::ProxyAdmin, "/key/generate", true)]
    #[case::admin_spend(RbacRole::ProxyAdmin, "/spend/logs", true)]
    #[case::admin_not_openai(RbacRole::ProxyAdmin, "/chat/completions", false)]
    #[case::team_openai(RbacRole::Team, "/chat/completions", true)]
    #[case::team_mcp(RbacRole::Team, "/mcp/tools/list", true)]
    #[case::team_exact(RbacRole::Team, "/v1/messages", true)]
    #[case::team_not_management(RbacRole::Team, "/key/generate", false)]
    #[case::internal_user_never(RbacRole::InternalUser, "/chat/completions", false)]
    fn default_allowed_routes(#[case] role: RbacRole, #[case] route: &str, #[case] expected: bool) {
        assert_eq!(
            allowed_routes_check(role, route, &JwtSettings::default()),
            expected
        );
    }

    #[rstest]
    #[case::listed(false, "/chat/completions", true)]
    #[case::pass_through_exempt(true, "/my-pass-through", true)]
    #[case::unlisted(false, "/my-pass-through", false)]
    fn team_route(
        #[case] auth_enforced_pass_through: bool,
        #[case] route: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(
            is_team_route_allowed(route, auth_enforced_pass_through, &JwtSettings::default()),
            expected
        );
    }

    #[test]
    fn admin_route_denial_lists_expanded_routes() {
        let settings = JwtSettings {
            config: JwtAuthSettings {
                admin_allowed_routes: strings(&["/a", "/b/*"]),
                ..JwtAuthSettings::default()
            },
            ..JwtSettings::default()
        };
        assert_eq!(check_admin_route("/b/c", &settings), Ok(()));
        assert_eq!(
            check_admin_route("/c", &settings).map_err(|denial| denial.public()),
            Err(crate::PublicDenial::Plain {
                message: "Admin not allowed to access this route. Route=/c, Allowed Routes=[\"/a\",\"/b/*\"]".to_owned()
            })
        );
    }

    #[test]
    fn admin_route_denial_expands_groups() {
        let settings = JwtSettings {
            config: JwtAuthSettings {
                admin_allowed_routes: strings(&["master_key_only_routes"]),
                ..JwtAuthSettings::default()
            },
            ..JwtSettings::default()
        };
        assert_eq!(
            check_admin_route("/c", &settings),
            Err(RbacDenial::AdminRoute {
                route: "/c".to_owned(),
                allowed: vec![
                    "/global/spend/reset".to_owned(),
                    "/memory-usage-in-mem-cache".to_owned(),
                    "/memory-usage-in-mem-cache-items".to_owned(),
                ],
            })
        );
    }

    /// The RFC 8693 exchange authorizes the IdP JWT against `POST /token`, so the default team
    /// allowlist must cover it or a multi-team claim never binds a team there.
    #[rstest]
    #[case::default_allowlist(None, true)]
    #[case::empty_allowlist(Some(vec![]), false)]
    fn team_role_reaches_the_token_endpoint_only_under_the_default_allowlist(
        #[case] team_allowed_routes: Option<Vec<String>>,
        #[case] allowed: bool,
    ) {
        let config = JwtAuthSettings::default();
        let settings = JwtSettings {
            config: JwtAuthSettings {
                team_allowed_routes: team_allowed_routes
                    .map_or_else(|| config.team_allowed_routes.clone(), Vec::into_boxed_slice),
                ..config
            },
            ..JwtSettings::default()
        };

        assert_eq!(
            allowed_routes_check(RbacRole::Team, "/token", &settings),
            allowed
        );
    }
}
