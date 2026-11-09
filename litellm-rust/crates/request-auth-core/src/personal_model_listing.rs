use crate::{Error, personal_model_request::Loaded, personal_models, personal_owner};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub owner: personal_owner::Facts,
    pub names: Option<Vec<String>>,
    pub models: Vec<Loaded>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    ReadOwner,
    ReadNames,
    ReadModel { index: usize, teams: Vec<usize> },
    Complete { selected: Vec<usize> },
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    let teams = match personal_owner::decide(&facts.owner)? {
        personal_owner::Decision::Complete { selected, .. } => selected,
        _ => return Ok(Decision::ReadOwner),
    };
    if facts
        .owner
        .teams
        .as_ref()
        .is_none_or(|teams| teams.loads.is_empty())
    {
        return Ok(Decision::Complete { selected: vec![] });
    }
    let Some(names) = &facts.names else {
        return Ok(Decision::ReadNames);
    };
    let mut selected = vec![];
    for index in 0..names.len() {
        let Some(loaded) = facts.models.get(index) else {
            return Ok(Decision::ReadModel { index, teams });
        };
        match personal_models::decide(personal_models::Request::Loaded {
            present: loaded.present.clone(),
            scopes: loaded.scopes.clone(),
        })? {
            personal_models::Decision::Complete { allowed } => {
                if allowed {
                    selected.push(index);
                }
            }
            _ => return Err(Error::InvalidAnswer),
        }
    }
    Ok(Decision::Complete { selected })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::missing_database(false, Decision::Complete { selected: vec![] })]
    #[case::load(true, Decision::ReadOwner)]
    fn owner_before_names(#[case] database_present: bool, #[case] expected: Decision) {
        let facts = Facts {
            owner: personal_owner::Facts {
                user_present: true,
                database_present,
                cache_present: true,
                logging_present: true,
                owner_loaded: false,
                owner_present: false,
                owner_failed: false,
                teams: None,
            },
            names: None,
            models: vec![],
        };
        assert_eq!(decide(&facts).unwrap(), expected);
    }
}
