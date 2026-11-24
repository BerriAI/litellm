use litellm_config::RbacRole;
use serde_json::Value;

use super::jwt::{AgentMode, HeaderTeam, ManagedAgent, Team, User};
use crate::{Rows, commit::Grants};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CallerKind {
    ProxyAdmin,
    Jwt,
    ManagedAgent(AgentMode),
}

/// What a JWT's claims say about who sends it, after the cheap agent reads and before any team,
/// user or organization is read.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Caller<R: Rows> {
    pub(crate) kind: CallerKind,
    pub(crate) rbac_role: Option<RbacRole>,
    pub(crate) scopes: Vec<String>,
    pub(crate) agent: Option<ManagedAgent<R::Agent>>,
    /// The managed agent's id, or the registered agent the agent claim names.
    pub(crate) agent_id: Option<String>,
    /// The `iss` claim, which a managed agent's issuer has to be trusted for.
    pub(crate) issuer: Value,
}

/// Everything identity read for a caller that is neither a proxy admin nor an autonomous agent.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Principal<R: Rows> {
    pub(crate) caller: Caller<R>,
    pub(crate) user_id: Option<String>,
    pub(crate) user_email: Option<String>,
    pub(crate) user: Option<User<R::User>>,
    pub(crate) team_id: Option<String>,
    pub(crate) team: Option<Team<R::Team>>,
    pub(crate) membership: Option<R::Membership>,
    /// The header team the team was taken from, which may still need the user's membership.
    pub(crate) header_team: Option<HeaderTeam>,
    pub(crate) org_id: Option<String>,
    pub(crate) org: Option<R::Org>,
    pub(crate) end_user_id: Option<String>,
    pub(crate) end_user: Option<R::EndUser>,
    pub(crate) grants: Grants<R>,
}
