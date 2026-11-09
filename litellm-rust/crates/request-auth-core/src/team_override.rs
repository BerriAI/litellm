use crate::{
    Error,
    fallback_access::{self, GroupedScope},
    model_access::{self, Decision as ModelDecision, Facts as ModelFacts, ObjectType},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Deserialize)]
pub struct Facts {
    pub team: Option<GroupedScope>,
    pub assigned_models: Option<Vec<String>>,
    pub assigned_scope: Option<ModelFacts>,
}

#[derive(Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Allow,
    Deny { denial: ModelDecision },
    ReadTeam,
    ReadTeamGroups,
    ReadAssignedGroups,
    ReadAssignedScope,
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    let Some(team) = &facts.team else {
        return Ok(Decision::ReadTeam);
    };
    let denial = fallback_access::grouped(team.clone(), true)?;
    match denial {
        ModelDecision::Allow => return Ok(Decision::Allow),
        ModelDecision::ReadGroups | ModelDecision::ReadGroupScope { .. } => {
            return Ok(Decision::ReadTeamGroups);
        }
        ModelDecision::Deny {
            object_type: ObjectType::Team,
            ..
        } => (),
        _ => return Ok(Decision::Deny { denial }),
    }
    let Some(assigned) = &facts.assigned_models else {
        return Ok(Decision::ReadAssignedGroups);
    };
    if assigned.is_empty() {
        return Ok(Decision::Deny { denial });
    }
    let Some(scope) = &facts.assigned_scope else {
        return Ok(Decision::ReadAssignedScope);
    };
    match model_access::check(scope)? {
        ModelDecision::Allow => Ok(Decision::Allow),
        ModelDecision::Deny { .. } => Ok(Decision::Deny { denial }),
        error => Ok(Decision::Deny { denial: error }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn team_is_read_before_override_groups() {
        assert!(matches!(
            decide(&Facts::default()).unwrap(),
            Decision::ReadTeam
        ));
    }
}
