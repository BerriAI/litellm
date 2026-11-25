use litellm_config::RbacRole;
use serde_json::Value;

use super::jwt::{AgentMode, HeaderTeam, JwtIdentity, ManagedAgent};
use crate::{Rows, commit::Grants};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClaimantKind {
    ProxyAdmin,
    Jwt,
    ManagedAgent(AgentMode),
}

/// What a JWT's claims say about who sends it, after the cheap agent reads and before any team,
/// user or organization is read.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Claimant<R: Rows> {
    pub(crate) kind: ClaimantKind,
    pub(crate) rbac_role: Option<RbacRole>,
    pub(crate) scopes: Vec<String>,
    pub(crate) agent: Option<ManagedAgent<R::Agent>>,
    /// The managed agent's id, or the registered agent the agent claim names.
    pub(crate) agent_id: Option<String>,
    /// The `iss` claim, which a managed agent's issuer has to be trusted for.
    pub(crate) issuer: Value,
}

/// The identity read for a claimant that is neither a proxy admin nor an autonomous agent, with
/// what the last route check and `commit` still need beside it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Principal<R: Rows> {
    pub(crate) identity: JwtIdentity<R>,
    /// The header team the team was taken from, which may still need the user's membership.
    pub(crate) header_team: Option<HeaderTeam>,
    /// The teams the user's row lists.
    pub(crate) user_teams: Vec<String>,
    pub(crate) grants: Grants<R>,
}
