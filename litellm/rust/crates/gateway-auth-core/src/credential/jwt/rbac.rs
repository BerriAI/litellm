use litellm_config::{RbacRole, RolePermissions, ScopeMapping};

use super::{JwtSettings, RbacDenial};
use crate::routes::RouteAllowlist;

/// A model list, which names each model it allows exactly.
#[derive(Clone, Copy, Debug)]
struct ModelAllowlist<'a>(&'a [String]);

impl<'a> ModelAllowlist<'a> {
    fn permits(self, model: &str) -> bool {
        self.0.iter().any(|allowed| allowed == model)
    }

    fn entries(self) -> &'a [String] {
        self.0
    }
}

impl JwtSettings {
    /// The first `role_permissions` entry for the role decides.
    fn permissions_of(&self, role: RbacRole) -> Option<&RolePermissions> {
        self.role_permissions
            .iter()
            .find(|permissions| permissions.role == role)
    }

    /// `get_role_based_models`
    fn role_models(&self, role: RbacRole) -> Option<ModelAllowlist<'_>> {
        self.permissions_of(role)?
            .models
            .as_deref()
            .map(ModelAllowlist)
    }

    /// `get_role_based_routes`
    fn role_routes(&self, role: RbacRole) -> Option<RouteAllowlist<'_>> {
        let routes = self.permissions_of(role)?.routes.as_deref()?;
        Some(self.route_groups.allowlist(routes))
    }

    /// `allowed_routes_check` for `PROXY_ADMIN`
    pub(super) fn admin_routes(&self) -> RouteAllowlist<'_> {
        self.route_groups
            .allowlist(&self.config.admin_allowed_routes)
    }

    /// `litellm_jwtauth.team_allowed_routes`
    pub(super) fn team_routes(&self) -> RouteAllowlist<'_> {
        self.route_groups
            .allowlist(&self.config.team_allowed_routes)
    }

    /// The gate `auth_builder` puts on `check_scope_based_access`: `enforce_scope_based_access` and
    /// a non-empty `scope_mappings`.
    pub(super) fn enforced_scope_mappings(&self) -> Option<&[ScopeMapping]> {
        self.config
            .scope_mappings
            .as_deref()
            .filter(|mappings| self.config.enforce_scope_based_access && !mappings.is_empty())
    }
}

/// `JWTAuthManager.can_rbac_role_call_route`
fn can_role_call_route(
    role: RbacRole,
    route: &str,
    settings: &JwtSettings,
) -> Result<(), RbacDenial> {
    let Some(allowed) = settings.role_routes(role) else {
        return Ok(());
    };
    if allowed.permits(route) {
        return Ok(());
    }
    Err(RbacDenial::RoleRoute {
        role,
        route: route.to_owned(),
        allowed: allowed.entries().into(),
    })
}

/// `JWTAuthManager.can_rbac_role_call_model`
fn can_role_call_model(
    role: RbacRole,
    model: Option<&str>,
    settings: &JwtSettings,
) -> Result<(), RbacDenial> {
    let (Some(allowed), Some(model)) = (settings.role_models(role), model) else {
        return Ok(());
    };
    if allowed.permits(model) {
        return Ok(());
    }
    Err(RbacDenial::RoleModel {
        role,
        model: model.to_owned(),
        allowed: allowed.entries().into(),
    })
}

/// `JWTAuthManager.check_scope_based_access` over [`JwtSettings::enforced_scope_mappings`]: the
/// models of every mapping whose scope the token has.
pub fn check_scope_based_access(
    mappings: &[ScopeMapping],
    scopes: &[String],
    model: Option<&str>,
) -> Result<(), RbacDenial> {
    let Some(model) = model.filter(|model| !model.is_empty()) else {
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
    if ModelAllowlist(&allowed).permits(model) {
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

/// The route half of `JWTAuthManager.check_admin_access`, for a token that has the admin scope.
pub fn check_admin_route(route: &str, settings: &JwtSettings) -> Result<(), RbacDenial> {
    let allowed = settings.admin_routes();
    if allowed.permits(route) {
        return Ok(());
    }
    Err(RbacDenial::AdminRoute {
        route: route.to_owned(),
        allowed: allowed.expanded(),
    })
}

/// `allowed_routes_check` for the team role: `team_allowed_routes` lists the route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TeamRouteListed(pub(super) bool);

/// `_is_team_route_allowed`: `team_allowed_routes` lists the route, or it is an auth-enforced
/// pass-through route, which the team's own grants gate instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TeamRouteAllowed(pub(super) bool);

pub fn team_route_listed(route: &str, settings: &JwtSettings) -> TeamRouteListed {
    TeamRouteListed(settings.team_routes().permits(route))
}

/// `auth_enforced_pass_through` is `RouteChecks.is_auth_enforced_pass_through_route(route, method)`,
/// which reads the host's registered pass-through endpoints.
pub fn is_team_route_allowed(
    route: &str,
    auth_enforced_pass_through: bool,
    settings: &JwtSettings,
) -> TeamRouteAllowed {
    TeamRouteAllowed(auth_enforced_pass_through || settings.team_routes().permits(route))
}

#[cfg(test)]
mod tests {
    use litellm_config::{JwtAuthSettings, RolePermissions, ScopeMapping};
    use rstest::rstest;

    use super::*;
    use crate::test_support::route_groups;

    fn defaults() -> JwtSettings {
        JwtSettings {
            route_groups: route_groups(),
            ..JwtSettings::default()
        }
    }

    fn strings(items: &[&str]) -> Box<[String]> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    fn scope_mappings() -> Box<[ScopeMapping]> {
        Box::new([
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
        ])
    }

    fn settings() -> JwtSettings {
        JwtSettings {
            config: JwtAuthSettings {
                enforce_rbac: true,
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
            ..defaults()
        }
    }

    #[rstest]
    #[case::allowed(RbacRole::Team, Some("gpt-a"), "/chat/completions", Ok(()))]
    #[case::group_template(RbacRole::Team, Some("gpt-a"), "/openai/deployments/my-model/chat/completions", Ok(()))]
    #[case::wildcard(RbacRole::Team, None, "/custom/x/y", Ok(()))]
    #[case::unlisted_role_is_unrestricted(RbacRole::ProxyAdmin, Some("anything"), "/key/generate", Ok(()))]
    #[case::role_without_lists(RbacRole::InternalUser, Some("anything"), "/key/generate", Ok(()))]
    #[case::empty_model_is_still_a_model(
        RbacRole::Team,
        Some(""),
        "/chat/completions",
        Err(RbacDenial::RoleModel { role: RbacRole::Team, model: String::new(), allowed: strings(&["gpt-a"]) })
    )]
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
            check_scope_based_access(&scope_mappings(), &scopes, model),
            expected
        );
    }

    #[rstest]
    #[case::enforced(true, Some(scope_mappings()), true)]
    #[case::not_enforced(false, Some(scope_mappings()), false)]
    #[case::no_mappings(true, None, false)]
    #[case::empty_mappings(true, Some(Box::default()), false)]
    fn scope_mappings_are_enforced_only_when_on_and_present(
        #[case] enforce_scope_based_access: bool,
        #[case] scope_mappings: Option<Box<[ScopeMapping]>>,
        #[case] enforced: bool,
    ) {
        let settings = JwtSettings {
            config: JwtAuthSettings {
                enforce_scope_based_access,
                scope_mappings,
                ..JwtAuthSettings::default()
            },
            ..defaults()
        };
        assert_eq!(settings.enforced_scope_mappings().is_some(), enforced);
    }

    #[rstest]
    #[case::management("/key/generate", true)]
    #[case::spend("/spend/logs", true)]
    #[case::not_openai("/chat/completions", false)]
    fn default_admin_routes(#[case] route: &str, #[case] expected: bool) {
        assert_eq!(defaults().admin_routes().permits(route), expected);
    }

    #[rstest]
    #[case::openai("/chat/completions", true)]
    #[case::mcp("/mcp/tools/list", true)]
    #[case::exact("/v1/messages", true)]
    #[case::not_management("/key/generate", false)]
    fn default_team_routes(#[case] route: &str, #[case] expected: bool) {
        assert_eq!(defaults().team_routes().permits(route), expected);
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
            is_team_route_allowed(route, auth_enforced_pass_through, &defaults()),
            TeamRouteAllowed(expected)
        );
    }

    #[test]
    fn admin_route_denial_lists_expanded_routes() {
        let settings = JwtSettings {
            config: JwtAuthSettings {
                admin_allowed_routes: strings(&["/a", "/b/*"]),
                ..JwtAuthSettings::default()
            },
            ..defaults()
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
            ..defaults()
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
            ..defaults()
        };

        assert_eq!(settings.team_routes().permits("/token"), allowed);
    }
}
