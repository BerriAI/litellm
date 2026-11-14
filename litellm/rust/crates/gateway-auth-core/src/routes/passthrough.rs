//! Grants for pass-through endpoints registered with `auth: true`. `RouteChecks` calls these through
//! `_native`, so the core and the Python key path share one implementation.

use super::{RouteAllowlist, matches_wildcard};

/// A `denied_passthrough_routes` entry that matched decides `Denied`; with none, `Denied` means
/// nothing granted the route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PassthroughAccess {
    Allowed,
    Denied { denied_route: Option<String> },
}

/// A team's `allowed_passthrough_routes` and `denied_passthrough_routes` metadata. `allowed` is
/// `None` when the key is missing or `None`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PassthroughGrants {
    pub allowed: Option<Vec<String>>,
    pub denied: Vec<String>,
}

/// `InitPassThroughEndpointHelpers.forwarded_routes(route)`: every route the request is forwarded as.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardedRoutes<'a>(pub &'a [String]);

/// A key's and its team's `allowed_passthrough_routes`.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllowedPassthrough<'a> {
    pub key: Option<&'a [String]>,
    pub team: Option<&'a [String]>,
}

impl AllowedPassthrough<'_> {
    /// `check_passthrough_route_access`: the key's list when it lists any, otherwise the team's.
    pub fn grants(&self, route: &str) -> bool {
        self.key
            .filter(|allowed| !allowed.is_empty())
            .or(self.team)
            .unwrap_or_default()
            .iter()
            .any(|allowed| matches_prefix(route, allowed))
    }
}

/// `denied_passthrough_routes` from a key's or a team's `metadata`
#[derive(Clone, Copy, Debug, Default)]
pub struct DeniedPassthrough<'a>(pub &'a [String]);

impl<'a> DeniedPassthrough<'a> {
    /// `matching_denied_passthrough_route`: the first entry that matches any route the request is
    /// forwarded as.
    pub fn first_match(&self, forwarded: ForwardedRoutes<'_>) -> Option<&'a str> {
        self.0.iter().map(String::as_str).find(|denied| {
            forwarded
                .0
                .iter()
                .any(|route| matches_denied(route, denied))
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PassthroughRequest<'a> {
    pub route: &'a str,
    /// `RouteChecks.is_auth_enforced_pass_through_route(route, method)`
    pub auth_enforced: bool,
    pub forwarded_routes: ForwardedRoutes<'a>,
    /// `litellm_jwtauth.team_allowed_routes`
    pub jwt_team_allowed_routes: Option<RouteAllowlist<'a>>,
}

impl PassthroughRequest<'_> {
    /// `team_has_passthrough_route_access`, naming the deny entry that matched when it refuses.
    pub fn team_access(&self, team: &PassthroughGrants) -> PassthroughAccess {
        if !self.auth_enforced {
            return PassthroughAccess::Allowed;
        }
        let denied_route = DeniedPassthrough(&team.denied).first_match(self.forwarded_routes);
        let team_grants = AllowedPassthrough {
            key: None,
            team: team.allowed.as_deref(),
        };
        if denied_route.is_none()
            && (self
                .jwt_team_allowed_routes
                .is_some_and(|routes| routes.grants_pass_through(self.route))
                || team_grants.grants(self.route))
        {
            return PassthroughAccess::Allowed;
        }
        PassthroughAccess::Denied {
            denied_route: denied_route.map(str::to_owned),
        }
    }
}

/// `_route_matches_allowed_route`: the route itself or anything under it, never a sibling that
/// only shares the prefix.
pub fn matches_prefix(route: &str, allowed_route: &str) -> bool {
    route
        .strip_prefix(allowed_route)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// `_route_matches_denied_route`. A `/` entry denies every route.
pub fn matches_denied(route: &str, denied_route: &str) -> bool {
    let normalized = denied_route.trim_end_matches('/');
    normalized.is_empty()
        || matches_prefix(route, normalized)
        || matches_wildcard(route, denied_route)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::test_support::route_groups;

    fn owned(routes: &[&str]) -> Vec<String> {
        routes.iter().map(|route| (*route).to_owned()).collect()
    }

    #[rstest]
    #[case::exact("/proxy-6", "/proxy-6", true)]
    #[case::child("/proxy-6/v1/chat", "/proxy-6", true)]
    #[case::sibling("/proxy-600", "/proxy-6", false)]
    #[case::shorter("/proxy", "/proxy-6", false)]
    fn prefix(#[case] route: &str, #[case] allowed: &str, #[case] expected: bool) {
        assert_eq!(matches_prefix(route, allowed), expected);
    }

    #[rstest]
    #[case::root_denies_all("/anything", "/", true)]
    #[case::trailing_slash("/openai/x", "/openai/", true)]
    #[case::wildcard("/openai/x", "/openai/*", true)]
    #[case::sibling("/openai-x", "/openai", false)]
    fn denied(#[case] route: &str, #[case] denied_route: &str, #[case] expected: bool) {
        assert_eq!(matches_denied(route, denied_route), expected);
    }

    #[rstest]
    #[case::first_entry_that_matches_any_forwarded_route(&["/a", "/v1/x"], &["/b", "/v1/*", "/a"], Some("/v1/*"))]
    #[case::none(&["/a"], &["/b"], None)]
    fn first_denied(
        #[case] forwarded: &[&str],
        #[case] denied: &[&str],
        #[case] expected: Option<&str>,
    ) {
        let (forwarded, denied) = (owned(forwarded), owned(denied));
        assert_eq!(
            DeniedPassthrough(&denied).first_match(ForwardedRoutes(&forwarded)),
            expected
        );
    }

    #[rstest]
    #[case::explicit(&["/openai/x"], true)]
    #[case::prefix_wildcard(&["/openai/*"], true)]
    #[case::blanket_star(&["*", "/*"], false)]
    #[case::route_group_name(&["openai_routes"], false)]
    fn jwt_team_routes(#[case] allowed: &[&str], #[case] expected: bool) {
        let (groups, allowed) = (route_groups(), owned(allowed));
        assert_eq!(
            groups.allowlist(&allowed).grants_pass_through("/openai/x"),
            expected
        );
    }

    #[rstest]
    #[case::key_list_wins(Some(&["/a"][..]), Some(&["/openai"][..]), false)]
    #[case::empty_key_list_falls_to_team(Some(&[][..]), Some(&["/openai"][..]), true)]
    #[case::team_only(None, Some(&["/openai"][..]), true)]
    #[case::neither(None, None, false)]
    fn allowed_routes(
        #[case] key: Option<&[&str]>,
        #[case] team: Option<&[&str]>,
        #[case] expected: bool,
    ) {
        let (key, team) = (key.map(owned), team.map(owned));
        let allowed = AllowedPassthrough {
            key: key.as_deref(),
            team: team.as_deref(),
        };
        assert_eq!(allowed.grants("/openai/x"), expected);
    }

    fn request<'a>(
        forwarded: &'a [String],
        jwt_team_allowed_routes: RouteAllowlist<'a>,
    ) -> PassthroughRequest<'a> {
        PassthroughRequest {
            route: "/openai/x",
            auth_enforced: true,
            forwarded_routes: ForwardedRoutes(forwarded),
            jwt_team_allowed_routes: Some(jwt_team_allowed_routes),
        }
    }

    fn grants(allowed: Option<&[&str]>, denied: &[&str]) -> PassthroughGrants {
        PassthroughGrants {
            allowed: allowed.map(owned),
            denied: owned(denied),
        }
    }

    #[rstest]
    #[case::team_grant(grants(Some(&["/openai"]), &[]), &[], PassthroughAccess::Allowed)]
    #[case::jwt_team_route_grant(grants(None, &[]), &["/openai/*"], PassthroughAccess::Allowed)]
    #[case::nothing_grants(grants(None, &[]), &[], PassthroughAccess::Denied { denied_route: None })]
    #[case::deny_beats_grant(
        grants(Some(&["/openai"]), &["/openai/*"]),
        &["/openai/*"],
        PassthroughAccess::Denied { denied_route: Some("/openai/*".to_owned()) }
    )]
    fn team_access(
        #[case] team: PassthroughGrants,
        #[case] jwt_team_allowed_routes: &[&str],
        #[case] expected: PassthroughAccess,
    ) {
        let (forwarded, allowed) = (owned(&["/openai/x"]), owned(jwt_team_allowed_routes));
        let groups = route_groups();
        assert_eq!(
            request(&forwarded, groups.allowlist(&allowed)).team_access(&team),
            expected
        );
    }

    #[test]
    fn routes_that_enforce_no_auth_are_always_allowed() {
        let request = PassthroughRequest {
            route: "/openai/x",
            ..PassthroughRequest::default()
        };
        assert_eq!(
            request.team_access(&grants(None, &["/"])),
            PassthroughAccess::Allowed
        );
    }
}
