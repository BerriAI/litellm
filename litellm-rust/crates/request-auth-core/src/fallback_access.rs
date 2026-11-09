use serde::{Deserialize, Serialize};

use crate::{
    Error,
    model_access::{self, Decision as ModelDecision, Facts as ModelFacts, Request, Response},
};

#[derive(Clone, Deserialize)]
pub struct GroupedScope {
    pub facts: ModelFacts,
    pub group_ids: Vec<String>,
    pub group_models: Option<Vec<String>>,
}

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub refuse_agent_keys: bool,
    pub agent_id: Option<String>,
    pub key: Option<GroupedScope>,
    pub team: Option<GroupedScope>,
    pub embedded_team: Option<ModelFacts>,
    pub assigned_models: Option<Vec<String>>,
    pub assigned_scope: Option<ModelFacts>,
    pub member: Option<ModelFacts>,
    pub project: Option<ModelFacts>,
    pub team_present: bool,
    pub embedded_team_grants: Vec<String>,
    pub member_grants: Vec<String>,
    pub project_grants: Vec<String>,
}

#[derive(Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Allow,
    Deny { denial: Option<ModelDecision> },
    ReadKey,
    ReadTeam,
    ReadEmbeddedTeam,
    ReadMember,
    ReadProject,
    ReadKeyGroups,
    ReadTeamGroups,
    ReadAssignedGroups,
    ReadAssignedScope,
}

pub fn grouped(scope: GroupedScope, merge: bool) -> Result<ModelDecision, Error> {
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
    let Some(key) = facts.key else {
        return Ok(Decision::ReadKey);
    };
    let key = grouped(key, false)?;
    if matches!(
        key,
        ModelDecision::ReadGroups | ModelDecision::ReadGroupScope { .. }
    ) {
        return Ok(Decision::ReadKeyGroups);
    }
    if !key.allowed() {
        return Ok(Decision::Deny { denial: Some(key) });
    }
    if facts.team_present && facts.team.is_none() {
        return Ok(Decision::ReadTeam);
    }
    if let Some(team) = facts.team {
        let decision = grouped(team, true)?;
        if matches!(
            decision,
            ModelDecision::ReadGroups | ModelDecision::ReadGroupScope { .. }
        ) {
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
            if grants.is_empty() {
                return Ok(Decision::Deny {
                    denial: Some(decision),
                });
            }
            let Some(assigned) = facts.assigned_scope else {
                return Ok(Decision::ReadAssignedScope);
            };
            if !model_access::check(&assigned)?.allowed() {
                return Ok(Decision::Deny {
                    denial: Some(decision),
                });
            }
        }
    } else if facts.embedded_team.is_none() && !facts.embedded_team_grants.is_empty() {
        return Ok(Decision::ReadEmbeddedTeam);
    } else if let Some(team) = facts.embedded_team {
        let decision = model_access::check(&team)?;
        if !decision.allowed() {
            return Ok(Decision::Deny {
                denial: Some(decision),
            });
        }
    }
    if facts.member.is_none() && !facts.member_grants.is_empty() {
        return Ok(Decision::ReadMember);
    }
    if let Some(scope) = facts.member {
        let decision = model_access::check(&scope)?;
        if !decision.allowed() {
            return Ok(Decision::Deny {
                denial: Some(decision),
            });
        }
    }
    if facts.project.is_none() && !facts.project_grants.is_empty() {
        return Ok(Decision::ReadProject);
    }
    if let Some(scope) = facts.project {
        let decision = model_access::check(&scope)?;
        if !decision.allowed() {
            return Ok(Decision::Deny {
                denial: Some(decision),
            });
        }
    }
    Ok(Decision::Allow)
}
