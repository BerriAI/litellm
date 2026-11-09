use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Deserialize, PartialEq, Eq, strum::VariantArray)]
pub enum Status {
    Found,
    Missing,
    Unavailable,
}

#[derive(Clone, Deserialize)]
pub struct Group {
    pub status: Status,
    pub models: Vec<String>,
    pub mcp_server_ids: Vec<String>,
    pub agent_ids: Vec<String>,
}

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub ids: Option<Vec<String>>,
    pub groups: Option<Vec<Group>>,
    pub check_db_only: bool,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    ReadIds,
    ReadGroups {
        authoritative: bool,
    },
    Complete {
        present: bool,
        ids: Vec<String>,
        models: BTreeSet<String>,
        mcp_server_ids: BTreeSet<String>,
        agent_ids: BTreeSet<String>,
        indices: Vec<usize>,
    },
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    let Some(ids) = &facts.ids else {
        return Ok(Decision::ReadIds);
    };
    let empty = |present| Decision::Complete {
        present,
        ids: vec![],
        models: BTreeSet::new(),
        mcp_server_ids: BTreeSet::new(),
        agent_ids: BTreeSet::new(),
        indices: vec![],
    };
    if ids.is_empty() {
        return Ok(empty(false));
    }
    let Some(groups) = &facts.groups else {
        return Ok(Decision::ReadGroups {
            authoritative: facts.check_db_only,
        });
    };
    if groups.len() != ids.len() {
        return Err(Error::InvalidAnswer);
    }
    let indices: Vec<_> = groups
        .iter()
        .enumerate()
        .filter_map(|(index, group)| (group.status == Status::Found).then_some(index))
        .collect();
    Ok(Decision::Complete {
        present: true,
        ids: ids
            .iter()
            .zip(groups)
            .filter(|(_, group)| group.status != Status::Missing)
            .map(|(id, _)| id.clone())
            .collect(),
        models: indices
            .iter()
            .flat_map(|index| groups[*index].models.iter().cloned())
            .collect(),
        mcp_server_ids: indices
            .iter()
            .flat_map(|index| groups[*index].mcp_server_ids.iter().cloned())
            .collect(),
        agent_ids: indices
            .iter()
            .flat_map(|index| groups[*index].agent_ids.iter().cloned())
            .collect(),
        indices,
    })
}

#[derive(Clone, Copy, Deserialize, strum::VariantArray)]
pub enum ReadStatus {
    Success,
    NotFound,
    HttpError,
}

#[derive(Clone, Deserialize)]
pub struct LoadFacts {
    pub database_present: bool,
    pub check_db_only: bool,
    pub status: Option<ReadStatus>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum LoadDecision {
    Read,
    Return,
    DatabaseUnavailable,
    HttpUnavailable,
    Raise,
}

pub fn load(facts: &LoadFacts) -> LoadDecision {
    if !facts.database_present {
        return LoadDecision::DatabaseUnavailable;
    }
    match facts.status {
        None => LoadDecision::Read,
        Some(ReadStatus::Success) => LoadDecision::Return,
        Some(ReadStatus::NotFound) => LoadDecision::Raise,
        Some(ReadStatus::HttpError) if facts.check_db_only => LoadDecision::Raise,
        Some(ReadStatus::HttpError) => LoadDecision::HttpUnavailable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::soft(false, LoadDecision::HttpUnavailable)]
    #[case::authoritative(true, LoadDecision::Raise)]
    fn http_lookup_policy(#[case] check_db_only: bool, #[case] expected: LoadDecision) {
        assert_eq!(
            load(&LoadFacts {
                database_present: true,
                check_db_only,
                status: Some(ReadStatus::HttpError)
            }),
            expected
        );
    }
}
