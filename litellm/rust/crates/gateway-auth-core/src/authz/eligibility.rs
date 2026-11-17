use crate::routes::passthrough::{PassthroughAccess, PassthroughGrants, PassthroughRequest};

/// The one place identity's team selection reads the route: whether `team_allowed_routes` lists
/// it, and what the pass-through grants of a candidate team allow.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TeamEligibility<'a> {
    /// `allowed_routes_check` for the team role: `team_allowed_routes` lists the route.
    pub(crate) route_listed: bool,
    /// `_is_team_route_allowed`: listed, or an auth-enforced pass-through route, which the team's
    /// own grants gate instead.
    pub(crate) route_allowed: bool,
    pub(crate) passthrough: PassthroughRequest<'a>,
}

/// What one of the token's teams contributes to `find_team_with_model_access`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    Selected,
    /// The route and pass-through grants allow it, the model does not.
    Fallback,
    RouteDenied,
    PassthroughDenied(Option<String>),
    /// It neither qualifies nor explains a denial.
    NoAccess,
}

impl TeamEligibility<'_> {
    pub(crate) fn route(&self) -> &str {
        self.passthrough.route
    }

    pub(crate) fn route_allowed(&self) -> bool {
        self.route_allowed
    }

    /// A team the token claims, given whether it may use the request's model.
    pub(crate) fn verdict(&self, grants: &PassthroughGrants, has_model: bool) -> Verdict {
        match (
            has_model,
            self.route_listed,
            self.passthrough.team_access(grants),
        ) {
            (true, true, PassthroughAccess::Allowed) => Verdict::Selected,
            (false, true, PassthroughAccess::Allowed) => Verdict::Fallback,
            (true, false, _) => Verdict::RouteDenied,
            (true, true, PassthroughAccess::Denied { denied_route }) => {
                Verdict::PassthroughDenied(denied_route)
            }
            (false, ..) => Verdict::NoAccess,
        }
    }

    /// A team the user belongs to that `_resolve_db_team_fallback` may fall back on whatever the
    /// model: the route is allowed and its grants let the request through.
    pub(crate) fn member_fallback(&self, grants: &PassthroughGrants) -> bool {
        self.route_allowed && self.passthrough.team_access(grants) == PassthroughAccess::Allowed
    }
}
