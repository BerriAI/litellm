use crate::{
    Error,
    fallback_access::{self, GroupedScope},
    member_models,
    model_access::{self, Decision as ModelDecision},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Default, Deserialize, PartialEq, Eq, strum::VariantArray)]
pub enum Purpose {
    #[default]
    All,
    Ceilings,
    Caller,
}

#[derive(Clone, Default, Deserialize)]
pub struct Facts {
    pub purpose: Purpose,
    pub model_present: bool,
    pub token_present: bool,
    pub managed_loaded: bool,
    pub managed_present: bool,
    pub managed_models_valid: bool,
    pub managed_models_count: usize,
    pub managed: Option<model_access::Facts>,
    pub ceilings: Option<Vec<Vec<String>>>,
    pub aliases_loaded: bool,
    pub scopes: Vec<model_access::Facts>,
    pub caller_loaded: bool,
    pub caller_present: bool,
    pub team_loaded: bool,
    pub team_present: bool,
    pub team: Option<GroupedScope>,
    pub member: Option<member_models::Facts>,
    pub user_loaded: bool,
    pub user_present: bool,
    pub user_grants: Vec<String>,
    pub user: Option<model_access::Facts>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Allow,
    Deny {
        denial: ModelDecision,
        member_models: Option<Vec<String>>,
    },
    ReadManaged,
    ReadManagedScope,
    ReadCeilings,
    ReadAliases,
    ReadCeilingScope {
        index: usize,
    },
    ReadCaller,
    ReadTeam,
    ReadTeamScope,
    ReadTeamGroups,
    ReadMember,
    ReadUser,
    ReadUserScope,
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    let deny = |denial| {
        Ok(Decision::Deny {
            denial,
            member_models: None,
        })
    };
    if facts.purpose == Purpose::All && !facts.managed_loaded {
        return Ok(Decision::ReadManaged);
    }
    if !facts.model_present || !facts.token_present {
        return Ok(Decision::Allow);
    }
    if facts.purpose == Purpose::All && facts.managed_present {
        if !facts.managed_models_valid || facts.managed_models_count == 0 {
            return deny(ModelDecision::NoAgentModels);
        }
        let Some(scope) = &facts.managed else {
            return Ok(Decision::ReadManagedScope);
        };
        let result = model_access::check(scope)?;
        if !result.allowed() {
            return deny(result);
        }
    }
    if facts.purpose != Purpose::Caller {
        let Some(ceilings) = &facts.ceilings else {
            return Ok(Decision::ReadCeilings);
        };
        if !facts.aliases_loaded {
            return Ok(Decision::ReadAliases);
        }
        for (index, models) in ceilings.iter().enumerate() {
            if models.is_empty() {
                return deny(ModelDecision::EmptyAgentCeiling);
            }
            let Some(scope) = facts.scopes.get(index) else {
                return Ok(Decision::ReadCeilingScope { index });
            };
            let result = model_access::check(scope)?;
            if !result.allowed() {
                return deny(result);
            }
        }
        if facts.purpose == Purpose::Ceilings {
            return Ok(Decision::Allow);
        }
    }
    if !facts.caller_loaded {
        return Ok(Decision::ReadCaller);
    }
    if !facts.caller_present {
        return Ok(Decision::Allow);
    }
    if !facts.team_loaded {
        return Ok(Decision::ReadTeam);
    }
    if facts.team_present {
        let Some(team) = &facts.team else {
            return Ok(Decision::ReadTeamScope);
        };
        let result = fallback_access::grouped(team.clone(), true)?;
        if matches!(
            result,
            ModelDecision::ReadGroups | ModelDecision::ReadGroupScope { .. }
        ) {
            return Ok(Decision::ReadTeamGroups);
        }
        if !result.allowed() {
            return deny(result);
        }
        let Some(member) = &facts.member else {
            return Ok(Decision::ReadMember);
        };
        return match member_models::decide(member)? {
            member_models::Decision::Allow => Ok(Decision::Allow),
            member_models::Decision::Deny {
                denial,
                member_models,
            } => Ok(Decision::Deny {
                denial,
                member_models,
            }),
            _ => Err(Error::InvalidAnswer),
        };
    }
    if !facts.user_loaded {
        return Ok(Decision::ReadUser);
    }
    let precheck = model_access::decide(model_access::Request::UserPrecheck {
        present: facts.user_present,
        grants: facts.user_grants.clone(),
    })?;
    let model_access::Response::Decision(precheck) = precheck else {
        return Err(Error::InvalidAnswer);
    };
    if !matches!(precheck, ModelDecision::ReadScope) {
        return if precheck.allowed() {
            Ok(Decision::Allow)
        } else {
            deny(precheck)
        };
    }
    let Some(user) = &facts.user else {
        return Ok(Decision::ReadUserScope);
    };
    let result = model_access::check(user)?;
    if result.allowed() {
        Ok(Decision::Allow)
    } else {
        deny(result)
    }
}

#[derive(Clone, Default, Deserialize)]
pub struct CeilingSource {
    pub agent_present: bool,
    pub managed_loaded: bool,
    pub managed_present: bool,
    pub loaded: bool,
    pub count: usize,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum SourceDecision {
    ReadManaged,
    ReadManagedCeilings,
    ReadUnmanagedCeiling,
    Complete { indices: Vec<usize> },
}

pub fn ceiling_source(facts: &CeilingSource) -> SourceDecision {
    if !facts.agent_present {
        return SourceDecision::Complete { indices: vec![] };
    }
    if !facts.managed_loaded {
        return SourceDecision::ReadManaged;
    }
    if facts.loaded {
        return SourceDecision::Complete {
            indices: (0..facts.count).collect(),
        };
    }
    if facts.managed_present {
        SourceDecision::ReadManagedCeilings
    } else {
        SourceDecision::ReadUnmanagedCeiling
    }
}

#[derive(Clone, Deserialize)]
pub struct GrantFacts {
    pub models: Option<Vec<Option<String>>>,
    pub ceilings: Vec<Vec<String>>,
}

pub fn grants(facts: &GrantFacts) -> std::collections::BTreeSet<String> {
    facts
        .models
        .iter()
        .flatten()
        .flatten()
        .chain(facts.ceilings.iter().flatten())
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::main(Purpose::All, Decision::ReadManaged)]
    #[case::groups(Purpose::Ceilings, Decision::Allow)]
    #[case::caller(Purpose::Caller, Decision::Allow)]
    fn empty_model_read_order(#[case] purpose: Purpose, #[case] expected: Decision) {
        assert_eq!(
            decide(&Facts {
                purpose,
                ..Facts::default()
            })
            .unwrap(),
            expected
        );
    }
}
