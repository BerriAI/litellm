use serde::{Deserialize, Serialize};

use crate::{
    Error,
    model_access::{self, Decision as ModelDecision, Facts as ModelFacts, Request, Response},
};

#[derive(Deserialize)]
pub struct GroupedScope {
    pub facts: ModelFacts,
    pub group_ids: Vec<String>,
    pub group_models: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct Facts {
    pub refuse_agent_keys: bool,
    pub agent_id: Option<String>,
    pub key: GroupedScope,
    pub team: Option<GroupedScope>,
    pub embedded_team: Option<ModelFacts>,
    pub assigned_models: Option<Vec<String>>,
    pub member: Option<ModelFacts>,
    pub project: Option<ModelFacts>,
}

#[derive(Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Allow,
    Deny { denial: Option<ModelDecision> },
    ReadKeyGroups,
    ReadTeamGroups,
    ReadAssignedGroups,
}

fn grouped(scope: GroupedScope, merge: bool) -> Result<ModelDecision, Error> {
    let Response::Decision(decision) = model_access::decide(Request::Grouped {
        facts: scope.facts,
        group_ids: scope.group_ids,
        group_models: scope.group_models,
        merge,
    })?
    else {
        return Err(Error::InvalidAnswer);
    };
    Ok(decision)
}

pub fn decide(facts: Facts) -> Result<Decision, Error> {
    if facts.refuse_agent_keys && facts.agent_id.is_some() {
        return Ok(Decision::Deny { denial: None });
    }
    let key = grouped(facts.key, false)?;
    if matches!(key, ModelDecision::ReadGroups) {
        return Ok(Decision::ReadKeyGroups);
    }
    if !key.allowed() {
        return Ok(Decision::Deny { denial: Some(key) });
    }
    if let Some(team) = facts.team {
        let model_facts = team.facts.clone();
        let decision = grouped(team, true)?;
        if matches!(decision, ModelDecision::ReadGroups) {
            return Ok(Decision::ReadTeamGroups);
        }
        if !decision.allowed() {
            if !matches!(decision, ModelDecision::Deny { .. }) {
                return Ok(Decision::Deny {
                    denial: Some(decision),
                });
            }
            let Some(grants) = facts.assigned_models else {
                return Ok(Decision::ReadAssignedGroups);
            };
            if grants.is_empty()
                || !model_access::check(&ModelFacts {
                    grants,
                    ..model_facts
                })?
                .allowed()
            {
                return Ok(Decision::Deny {
                    denial: Some(decision),
                });
            }
        }
    } else if let Some(team) = facts.embedded_team {
        let decision = model_access::check(&team)?;
        if !decision.allowed() {
            return Ok(Decision::Deny {
                denial: Some(decision),
            });
        }
    }
    for scope in [facts.member, facts.project].into_iter().flatten() {
        let decision = model_access::check(&scope)?;
        if !decision.allowed() {
            return Ok(Decision::Deny {
                denial: Some(decision),
            });
        }
    }
    Ok(Decision::Allow)
}
