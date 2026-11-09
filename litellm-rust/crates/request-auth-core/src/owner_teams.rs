use crate::{Error, personal_budget};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Deserialize, PartialEq, Eq, strum::VariantArray)]
pub enum Purpose {
    Models,
    Grants,
}

#[derive(Clone, Deserialize)]
pub struct Load {
    pub present: bool,
    pub failed: bool,
    pub blocked: bool,
    pub org: Option<String>,
}

#[derive(Clone, Deserialize)]
pub struct Budget {
    pub index: usize,
    pub facts: personal_budget::Facts,
}

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub purpose: Purpose,
    pub user_present: bool,
    pub database_present: bool,
    pub ids: Vec<String>,
    pub ui_team_id: String,
    pub org: Option<String>,
    pub loaded: bool,
    pub loads: Vec<Load>,
    pub budgets: Vec<Budget>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    ReadTeams {
        ids: Vec<String>,
    },
    ReadBudgets {
        indices: Vec<usize>,
    },
    Complete {
        selected: Vec<usize>,
        withheld: Vec<usize>,
        unreadable: bool,
    },
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    if facts.purpose == Purpose::Grants && (!facts.user_present || !facts.database_present) {
        return Ok(Decision::Complete {
            selected: vec![],
            withheld: vec![],
            unreadable: false,
        });
    }
    if !facts.loaded {
        let mut seen = BTreeSet::new();
        let ids = facts
            .ids
            .iter()
            .filter(|id| {
                facts.purpose == Purpose::Models || (!id.is_empty() && **id != facts.ui_team_id)
            })
            .filter(|id| seen.insert(*id))
            .cloned()
            .collect();
        return Ok(Decision::ReadTeams { ids });
    }
    let eligible: Vec<_> = facts
        .loads
        .iter()
        .enumerate()
        .filter(|(_, load)| load.present && !load.blocked)
        .filter(|(_, load)| {
            facts.purpose == Purpose::Models
                || facts.org.is_none()
                || load.org.is_none()
                || load.org == facts.org
        })
        .map(|(index, _)| index)
        .collect();
    let unreadable = facts.loads.iter().any(|load| load.failed);
    if facts.purpose == Purpose::Models {
        return Ok(Decision::Complete {
            selected: eligible,
            withheld: vec![],
            unreadable,
        });
    }
    let missing: Vec<_> = eligible
        .iter()
        .filter(|index| !facts.budgets.iter().any(|budget| budget.index == **index))
        .copied()
        .collect();
    if !missing.is_empty() {
        return Ok(Decision::ReadBudgets { indices: missing });
    }
    let results: Result<Vec<_>, _> = eligible
        .iter()
        .map(|index| {
            let budget = facts
                .budgets
                .iter()
                .find(|budget| budget.index == *index)
                .ok_or(Error::InvalidAnswer)?;
            match personal_budget::decide(&budget.facts)? {
                personal_budget::Decision::Complete { allowed, .. } => Ok((*index, allowed)),
                _ => Err(Error::InvalidAnswer),
            }
        })
        .collect();
    let results = results?;
    let blocked = facts
        .loads
        .iter()
        .enumerate()
        .filter(|(_, load)| {
            load.present
                && load.blocked
                && (facts.org.is_none() || load.org.is_none() || load.org == facts.org)
        })
        .map(|(index, _)| index);
    let withheld = blocked
        .chain(
            results
                .iter()
                .filter_map(|(index, allowed)| (!allowed).then_some(*index)),
        )
        .collect();
    Ok(Decision::Complete {
        selected: results
            .into_iter()
            .filter_map(|(index, allowed)| allowed.then_some(index))
            .collect(),
        withheld,
        unreadable,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::models(Purpose::Models, vec!["", "ui", "a"])]
    #[case::grants(Purpose::Grants, vec!["a"])]
    fn id_filter(#[case] purpose: Purpose, #[case] expected: Vec<&str>) {
        let facts = Facts {
            purpose,
            user_present: true,
            database_present: true,
            ids: vec!["".into(), "ui".into(), "a".into(), "a".into()],
            ui_team_id: "ui".into(),
            org: None,
            loaded: false,
            loads: vec![],
            budgets: vec![],
        };
        assert_eq!(
            decide(&facts).unwrap(),
            Decision::ReadTeams {
                ids: expected.into_iter().map(str::to_owned).collect()
            }
        );
    }
}
