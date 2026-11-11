//! Grants for pass-through endpoints registered with `auth: true`. `RouteChecks` calls these through
//! `_native`, so the core and the Python key path share one implementation.

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

#[derive(Clone, Copy, Debug, Default)]
pub struct PassthroughRequest<'a> {
    pub route: &'a str,
    /// `RouteChecks.is_auth_enforced_pass_through_route(route, method)`
    pub auth_enforced: bool,
    /// `InitPassThroughEndpointHelpers.forwarded_routes(route)`
    pub forwarded_routes: &'a [String],
    /// `litellm_jwtauth.team_allowed_routes`
    pub jwt_team_allowed_routes: &'a [String],
}

impl PassthroughRequest<'_> {
    /// `team_has_passthrough_route_access`, naming the deny entry that matched when it refuses.
    pub fn team_access(&self, team: &PassthroughGrants) -> PassthroughAccess {
        if !self.auth_enforced {
            return PassthroughAccess::Allowed;
        }
        let denied_route = first_denied_route(self.forwarded_routes, &team.denied);
        if denied_route.is_none()
            && (jwt_team_routes_grant(self.route, self.jwt_team_allowed_routes)
                || allowed_routes_grant(self.route, None, team.allowed.as_deref()))
        {
            return PassthroughAccess::Allowed;
        }
        PassthroughAccess::Denied {
            denied_route: denied_route.map(str::to_owned),
        }
    }
}

/// `route_matches_wildcard_pattern`: a trailing `*` matches by prefix, anything else exactly.
pub fn matches_wildcard(route: &str, pattern: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => route.starts_with(prefix),
        None => route == pattern,
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

/// `matching_denied_passthrough_route`: the first deny entry that matches any route the request
/// is forwarded as.
pub fn first_denied_route<'a, S: AsRef<str>>(
    forwarded_routes: &[String],
    denied_routes: &'a [S],
) -> Option<&'a str> {
    denied_routes.iter().map(AsRef::as_ref).find(|denied| {
        forwarded_routes
            .iter()
            .any(|route| matches_denied(route, denied))
    })
}

/// `jwt_team_routes_grant_pass_through`: explicit paths and trailing-wildcard prefixes grant,
/// blanket entries (`*`, `/*`) never do.
pub fn jwt_team_routes_grant<S: AsRef<str>>(route: &str, team_allowed_routes: &[S]) -> bool {
    team_allowed_routes
        .iter()
        .map(AsRef::as_ref)
        .filter(|allowed| !allowed.trim_end_matches('*').trim_matches('/').is_empty())
        .any(|allowed| matches_wildcard(route, allowed))
}

/// `check_passthrough_route_access`: the key's `allowed_passthrough_routes` when it lists any,
/// otherwise the team's.
pub fn allowed_routes_grant<S: AsRef<str>>(
    route: &str,
    key: Option<&[S]>,
    team: Option<&[S]>,
) -> bool {
    key.filter(|allowed| !allowed.is_empty())
        .or(team)
        .unwrap_or_default()
        .iter()
        .any(|allowed| matches_prefix(route, allowed.as_ref()))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn owned(routes: &[&str]) -> Vec<String> {
        routes.iter().map(|route| (*route).to_owned()).collect()
    }

    #[rstest]
    #[case::exact("/scim/v2", "/scim/v2", true)]
    #[case::wildcard_child("/scim/v2/Users/1", "/scim/v2/*", true)]
    #[case::wildcard_other("/chat/completions", "/scim/v2/*", false)]
    #[case::no_wildcard_child("/scim/v2/Users", "/scim/v2", false)]
    fn wildcard(#[case] route: &str, #[case] pattern: &str, #[case] expected: bool) {
        assert_eq!(matches_wildcard(route, pattern), expected);
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
        assert_eq!(first_denied_route(&owned(forwarded), denied), expected);
    }

    #[rstest]
    #[case::explicit(&["/openai/x"], true)]
    #[case::prefix_wildcard(&["/openai/*"], true)]
    #[case::blanket_star(&["*", "/*"], false)]
    #[case::route_group_name(&["openai_routes"], false)]
    fn jwt_team_routes(#[case] allowed: &[&str], #[case] expected: bool) {
        assert_eq!(jwt_team_routes_grant("/openai/x", allowed), expected);
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
        assert_eq!(allowed_routes_grant("/openai/x", key, team), expected);
    }

    fn request<'a>(
        forwarded: &'a [String],
        jwt_team_allowed_routes: &'a [String],
    ) -> PassthroughRequest<'a> {
        PassthroughRequest {
            route: "/openai/x",
            auth_enforced: true,
            forwarded_routes: forwarded,
            jwt_team_allowed_routes,
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
        assert_eq!(request(&forwarded, &allowed).team_access(&team), expected);
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
