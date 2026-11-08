use crate::{Error, owner_teams, personal_models};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
pub struct Loaded {
    pub present: Vec<bool>,
    pub scopes: Vec<personal_models::Scope>,
}

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub requested: Vec<String>,
    pub owner_present: bool,
    pub teams: Option<owner_teams::Facts>,
    pub models: Vec<Loaded>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Allow,
    Deny { model: String },
    ReadTeams,
    ReadModel { index: usize, teams: Vec<usize> },
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    let teams = if facts.owner_present {
        let Some(teams) = &facts.teams else {
            return Ok(Decision::ReadTeams);
        };
        match owner_teams::decide(teams)? {
            owner_teams::Decision::Complete { selected, .. } => selected,
            _ => return Err(Error::InvalidAnswer),
        }
    } else {
        vec![]
    };
    for (index, model) in facts.requested.iter().enumerate() {
        if !facts.owner_present {
            return Ok(Decision::Deny {
                model: model.clone(),
            });
        }
        let Some(loaded) = facts.models.get(index) else {
            return Ok(Decision::ReadModel { index, teams });
        };
        match personal_models::decide(personal_models::Request::Loaded {
            present: loaded.present.clone(),
            scopes: loaded.scopes.clone(),
        })? {
            personal_models::Decision::Complete { allowed: true } => (),
            personal_models::Decision::Complete { allowed: false } => {
                return Ok(Decision::Deny {
                    model: model.clone(),
                });
            }
            _ => return Err(Error::InvalidAnswer),
        }
    }
    Ok(Decision::Allow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::empty(false, vec![], Decision::Allow)]
    #[case::owner_lookup(true, vec![], Decision::ReadTeams)]
    #[case::absent(false, vec!["model".into()], Decision::Deny { model: "model".into() })]
    fn owner_and_empty_requests(
        #[case] owner_present: bool,
        #[case] requested: Vec<String>,
        #[case] expected: Decision,
    ) {
        let facts = Facts {
            requested,
            owner_present,
            teams: None,
            models: vec![],
        };
        assert_eq!(decide(&facts).unwrap(), expected);
    }
}
