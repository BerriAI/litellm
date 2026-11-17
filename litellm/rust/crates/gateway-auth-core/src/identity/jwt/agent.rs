use serde_json::Value;

use super::{DelegatedTeam, JwtIdentity, JwtRows, ManagedAgent, team::ClaimedTeam};
use crate::{AuthHost, Error, Rows, authn::jwt::JwtSettings, host::AgentRow};

/// An autonomous managed agent acts as itself, with no user, team or organization.
pub(crate) fn autonomous_identity<R: Rows>(
    claims: Value,
    agent: ManagedAgent<R::Agent>,
) -> JwtIdentity<R> {
    JwtIdentity {
        claims,
        is_proxy_admin: false,
        user_id: None,
        user_email: None,
        team_id: None,
        org_id: None,
        end_user_id: None,
        agent_id: Some(agent.agent_id.clone()),
        managed_agent: Some(agent),
        rows: JwtRows::default(),
    }
}

/// `resolve_delegated_agent_team` for a managed agent. Any other token keeps the team it claimed.
pub(crate) async fn delegated_team<H: AuthHost>(
    host: &H,
    agent: Option<&ManagedAgent<AgentRow<H>>>,
    claimed: &ClaimedTeam<H>,
    settings: &JwtSettings,
) -> Result<Option<String>, Error> {
    let Some(agent) = agent else {
        return Ok(claimed.team_id.clone());
    };
    let delegated = DelegatedTeam {
        user_id: agent.user_id.as_deref(),
        agent_id: &agent.agent_id,
        team_id: claimed.team_id.as_deref(),
        explicit_team: claimed.header_team.is_some(),
        allowed_team_ids: claimed
            .scope
            .as_ref()
            .filter(|_| !settings.config.fallback_to_db_teams),
    };
    host.delegated_agent_team(&delegated).await
}
