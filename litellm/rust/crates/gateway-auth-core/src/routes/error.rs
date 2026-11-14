use super::RouteGroup;

/// A settings build failure, never a request denial.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the host's route table has no {} group", <&str>::from(*group))]
pub struct MissingRouteGroup {
    pub group: RouteGroup,
}
