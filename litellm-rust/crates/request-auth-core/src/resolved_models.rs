use crate::{
    Error,
    fallback_access::{self, GroupedScope},
    member_models,
    model_access::{self, Decision as ModelDecision},
    personal_model_request, team_override,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
pub struct TeamRead {
    pub found: bool,
    pub failed: bool,
}

#[derive(Clone, Default, Deserialize)]
pub struct Facts {
    pub model: String,
    pub config_nonempty: bool,
    pub key_models: Vec<String>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    pub project_present: bool,
    pub customer_present: bool,
    pub database_present: bool,
    pub personal_enabled: bool,
    pub master: bool,
    pub role: Option<String>,
    pub key: Option<GroupedScope>,
    pub team_read: Option<TeamRead>,
    pub embedded_ready: bool,
    pub team: Option<team_override::Facts>,
    pub member: Option<member_models::Facts>,
    pub owner_loaded: bool,
    pub owner_present: bool,
    pub owner_failed: bool,
    pub personal: Option<personal_model_request::Facts>,
    pub project_loaded: bool,
    pub project_grants: Vec<String>,
    pub project: Option<model_access::Facts>,
    pub customer_loaded: bool,
    pub customer_grants: Vec<String>,
    pub customer: Option<model_access::Facts>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Allow,
    Deny {
        denial: ModelDecision,
        member_models: Option<Vec<String>>,
    },
    ReadKey,
    ReadKeyGroups,
    ReadTeamObject,
    ReadEmbeddedTeam,
    ReadTeamScope,
    ReadMember,
    ReadOwner,
    ReadPersonal,
    ReadProjectObject,
    ReadProjectScope,
    ReadCustomerObject,
    ReadCustomerScope,
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    let deny = |denial| {
        Ok(Decision::Deny {
            denial,
            member_models: None,
        })
    };
    if !facts.config_nonempty
        && !facts
            .key_models
            .iter()
            .any(|model| model == "all-team-models")
    {
        let Some(key) = &facts.key else {
            return Ok(Decision::ReadKey);
        };
        let key = fallback_access::grouped(key.clone(), false)?;
        if matches!(
            key,
            ModelDecision::ReadGroups | ModelDecision::ReadGroupScope { .. }
        ) {
            return Ok(Decision::ReadKeyGroups);
        }
        if !key.allowed() {
            return deny(key);
        }
    }
    if facts.team_id.is_some() {
        let Some(read) = &facts.team_read else {
            return Ok(Decision::ReadTeamObject);
        };
        if read.failed && !facts.embedded_ready {
            return Ok(Decision::ReadEmbeddedTeam);
        }
        if read.found || read.failed {
            let Some(team) = &facts.team else {
                return Ok(Decision::ReadTeamScope);
            };
            match team_override::decide(team)? {
                team_override::Decision::Allow => (),
                team_override::Decision::Deny { denial } => return deny(denial),
                _ => return Err(Error::InvalidAnswer),
            }
            if read.found && facts.user_id.is_some() {
                let Some(member) = &facts.member else {
                    return Ok(Decision::ReadMember);
                };
                match member_models::decide(member)? {
                    member_models::Decision::Allow => (),
                    member_models::Decision::Deny {
                        denial,
                        member_models,
                    } => {
                        return Ok(Decision::Deny {
                            denial,
                            member_models,
                        });
                    }
                    _ => return Err(Error::InvalidAnswer),
                }
            }
        }
    }
    let model_access::Response::Predicate(personal) =
        model_access::decide(model_access::Request::PersonalApplies {
            enabled: facts.personal_enabled,
            token_present: true,
            team_id: facts.team_id.clone(),
            user_id: facts.user_id.clone(),
            master: facts.master,
            role: facts.role.clone(),
            api_key_present: true,
            inheritance: false,
        })?
    else {
        return Err(Error::InvalidAnswer);
    };
    if personal {
        if !facts.owner_loaded {
            return Ok(Decision::ReadOwner);
        }
        if facts.owner_failed || !facts.owner_present {
            return deny(ModelDecision::PersonalTeam {
                model: facts.model.clone(),
            });
        }
        let Some(personal) = &facts.personal else {
            return Ok(Decision::ReadPersonal);
        };
        match personal_model_request::decide(personal)? {
            personal_model_request::Decision::Allow => (),
            personal_model_request::Decision::Deny { model } => {
                return deny(ModelDecision::PersonalTeam { model });
            }
            _ => return Err(Error::InvalidAnswer),
        }
    }
    if facts.project_present {
        if !facts.project_loaded {
            return Ok(Decision::ReadProjectObject);
        }
        if !facts.project_grants.is_empty() {
            let Some(project) = &facts.project else {
                return Ok(Decision::ReadProjectScope);
            };
            let decision = model_access::check(project)?;
            if !decision.allowed() {
                return deny(decision);
            }
        }
    }
    if facts.customer_present && facts.database_present {
        if !facts.customer_loaded {
            return Ok(Decision::ReadCustomerObject);
        }
        if !facts.customer_grants.is_empty() {
            let Some(customer) = &facts.customer else {
                return Ok(Decision::ReadCustomerScope);
            };
            let model_access::Response::Decision(decision) =
                model_access::decide(model_access::Request::Customer(customer.clone()))?
            else {
                return Err(Error::InvalidAnswer);
            };
            if !decision.allowed() {
                return deny(decision);
            }
        }
    }
    Ok(Decision::Allow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::key(false, Decision::ReadKey)]
    #[case::team(true, Decision::ReadTeamObject)]
    fn key_before_team(#[case] config_nonempty: bool, #[case] expected: Decision) {
        assert_eq!(
            decide(&Facts {
                config_nonempty,
                team_id: Some("team".into()),
                ..Facts::default()
            })
            .unwrap(),
            expected
        );
    }
}
