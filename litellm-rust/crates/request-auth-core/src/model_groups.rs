use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct Group {
    pub assigned_team_ids: Vec<String>,
    pub assigned_key_ids: Vec<String>,
    pub resources: Vec<String>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", content = "group")]
pub enum Lookup {
    Found(Group),
    Missing,
    Failed,
}

#[derive(Deserialize)]
pub struct Facts {
    pub subject_present: bool,
    pub database_present: bool,
    pub cache_present: bool,
    pub group_ids: Vec<String>,
    pub team_id: Option<String>,
    pub key_id: Option<String>,
    pub check_db_only: bool,
    pub strict_errors: bool,
    pub loaded: Vec<Lookup>,
}

#[derive(Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Read { index: usize },
    Raise { index: usize },
    Complete { resources: Vec<String> },
}

pub fn decide(facts: Facts) -> Decision {
    if !facts.subject_present || !facts.database_present || !facts.cache_present {
        return Decision::Complete { resources: vec![] };
    }
    let mut resources = BTreeSet::new();
    for index in 0..facts.group_ids.len() {
        let Some(loaded) = facts.loaded.get(index) else {
            return Decision::Read { index };
        };
        match loaded {
            Lookup::Missing | Lookup::Failed if facts.check_db_only || facts.strict_errors => {
                return Decision::Raise { index };
            }
            Lookup::Missing | Lookup::Failed => {}
            Lookup::Found(group) => {
                if facts
                    .team_id
                    .as_ref()
                    .is_some_and(|id| !id.is_empty() && group.assigned_team_ids.contains(id))
                    || facts
                        .key_id
                        .as_ref()
                        .is_some_and(|id| !id.is_empty() && group.assigned_key_ids.contains(id))
                {
                    resources.extend(group.resources.iter().cloned());
                }
            }
        }
    }
    Decision::Complete {
        resources: resources.into_iter().collect(),
    }
}
