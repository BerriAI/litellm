use crate::{Error, owner_teams};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub user_present: bool,
    pub database_present: bool,
    pub cache_present: bool,
    pub logging_present: bool,
    pub owner_loaded: bool,
    pub owner_present: bool,
    pub owner_failed: bool,
    pub teams: Option<owner_teams::Facts>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    ReadOwner,
    ReadTeams,
    Complete {
        selected: Vec<usize>,
        withheld: Vec<usize>,
        unreadable: bool,
    },
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    let empty = || {
        Ok(Decision::Complete {
            selected: vec![],
            withheld: vec![],
            unreadable: false,
        })
    };
    if !facts.user_present
        || !facts.database_present
        || !facts.cache_present
        || !facts.logging_present
    {
        return empty();
    }
    if !facts.owner_loaded {
        return Ok(Decision::ReadOwner);
    }
    if facts.owner_failed || !facts.owner_present {
        return empty();
    }
    let Some(teams) = &facts.teams else {
        return Ok(Decision::ReadTeams);
    };
    match owner_teams::decide(teams)? {
        owner_teams::Decision::Complete {
            selected,
            withheld,
            unreadable,
        } => Ok(Decision::Complete {
            selected,
            withheld,
            unreadable,
        }),
        _ => Err(Error::InvalidAnswer),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::no_database(false, Decision::Complete { selected: vec![], withheld: vec![], unreadable: false })]
    #[case::lookup(true, Decision::ReadOwner)]
    fn prerequisites(#[case] database_present: bool, #[case] expected: Decision) {
        let facts = Facts {
            user_present: true,
            database_present,
            cache_present: true,
            logging_present: true,
            owner_loaded: false,
            owner_present: false,
            owner_failed: false,
            teams: None,
        };
        assert_eq!(decide(&facts).unwrap(), expected);
    }
}
