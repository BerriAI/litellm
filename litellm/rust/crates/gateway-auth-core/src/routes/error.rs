use super::RouteGroup;

/// A route the proxy's own allowlist leaves out, refused before any credential is read.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("Access forbidden: Route {route} not allowed")]
pub struct RouteNotExposed {
    pub route: String,
}

/// A settings build failure, never a request denial.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the host's route table has no {} group", <&str>::from(*group))]
pub struct MissingRouteGroup {
    pub group: RouteGroup,
}
