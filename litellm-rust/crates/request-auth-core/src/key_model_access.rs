use serde::{Deserialize, Serialize};

use crate::{
    Error,
    fallback_access::{GroupedScope, grouped},
    model_access::Decision as ModelDecision,
};

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub config_nonempty: bool,
    pub key_models: Vec<String>,
    pub primary_loaded: bool,
    pub primary: Option<GroupedScope>,
    pub fallback_names: Option<Vec<String>>,
    pub fallback_scopes: Vec<GroupedScope>,
    pub routing: Vec<Routing>,
}

#[derive(Clone, Deserialize)]
pub enum Routing {
    Routed,
    Failed,
}

#[derive(Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Allow,
    Deny { denial: ModelDecision },
    ReadPrimary,
    ReadPrimaryGroups,
    ReadFallbackNames,
    ReadFallbackScope { index: usize },
    ReadFallbackGroups { index: usize },
    RouteFallback { index: usize },
    RoutingFailure { index: usize },
}

pub fn decide(facts: Facts) -> Result<Decision, Error> {
    if facts.config_nonempty
        || facts
            .key_models
            .iter()
            .any(|model| model == "all-team-models")
    {
        return Ok(Decision::Allow);
    }
    if !facts.primary_loaded {
        return Ok(Decision::ReadPrimary);
    }
    if let Some(primary) = facts.primary {
        let decision = grouped(primary, false)?;
        if matches!(
            decision,
            ModelDecision::ReadGroups | ModelDecision::ReadGroupScope { .. }
        ) {
            return Ok(Decision::ReadPrimaryGroups);
        }
        if !decision.allowed() {
            return Ok(Decision::Deny { denial: decision });
        }
    }
    let Some(names) = facts.fallback_names else {
        return Ok(Decision::ReadFallbackNames);
    };
    for index in 0..names.len() {
        let Some(scope) = facts.fallback_scopes.get(index) else {
            return Ok(Decision::ReadFallbackScope { index });
        };
        let decision = grouped(scope.clone(), false)?;
        if matches!(
            decision,
            ModelDecision::ReadGroups | ModelDecision::ReadGroupScope { .. }
        ) {
            return Ok(Decision::ReadFallbackGroups { index });
        }
        if !decision.allowed() {
            return Ok(Decision::Deny { denial: decision });
        }
        match facts.routing.get(index) {
            None => return Ok(Decision::RouteFallback { index }),
            Some(Routing::Failed) => return Ok(Decision::RoutingFailure { index }),
            Some(Routing::Routed) => {}
        }
    }
    Ok(Decision::Allow)
}
