use serde_json::Value;

use super::{AgentMode, claims};
use crate::{
    AuthHost, Error, JwtDenial,
    authn::jwt::{Claims, JwtSettings},
    identity::{Claimant, ClaimantKind},
};

/// `JWTAuthManager.resolve_agent_id`
pub(crate) async fn resolve_agent_id(
    host: &impl AuthHost,
    claims: &Claims,
    settings: &JwtSettings,
) -> Result<Option<String>, Error> {
    let Some(claim) = claims::agent_claim(claims, settings) else {
        return Ok(None);
    };
    match host.registered_agent(&claim).await? {
        Some(agent)
            if !agent.identity_managed && !agent.has_identity && !agent.has_legacy_identity =>
        {
            Ok(Some(agent.agent_id))
        }
        _ => Err(JwtDenial::UnregisteredAgent {
            agent_id_jwt_field: settings
                .config
                .agent_id_jwt_field
                .as_ref()
                .map(ToString::to_string),
            claim,
        }
        .into()),
    }
}

/// The claimant a JWT's claims name: a managed agent, a proxy admin or anyone else, with the role
/// and scopes the route policy decides on. An autonomous agent's other claims are never read.
pub(crate) async fn claimant<H: AuthHost>(
    host: &H,
    claims: &Claims,
    settings: &JwtSettings,
) -> Result<Claimant<H::Rows>, Error> {
    let issuer = claims.0.get("iss").cloned().unwrap_or(Value::Null);
    let agent = host.managed_agent(&Value::Object(claims.0.clone())).await?;
    if let Some(agent) = agent {
        let kind = ClaimantKind::ManagedAgent(agent.mode);
        let scopes = match agent.mode {
            AgentMode::Autonomous => Vec::new(),
            AgentMode::Delegated => claims::scopes(claims)?,
        };
        return Ok(Claimant {
            kind,
            rbac_role: None,
            scopes,
            agent_id: Some(agent.agent_id.clone()),
            agent: Some(agent),
            issuer,
        });
    }
    let rbac_role = claims::rbac_role(claims, settings)?;
    let scopes = claims::scopes(claims)?;
    let kind = match claims::is_admin(&scopes, settings) {
        true => ClaimantKind::ProxyAdmin,
        false => ClaimantKind::Jwt,
    };
    let agent_id = resolve_agent_id(host, claims, settings).await?;
    Ok(Claimant {
        kind,
        rbac_role,
        scopes,
        agent: None,
        agent_id,
        issuer,
    })
}
