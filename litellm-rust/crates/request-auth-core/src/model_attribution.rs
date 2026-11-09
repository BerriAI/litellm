use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Deserialize)]
pub struct Facts {
    pub model_present: bool,
    pub token_present: bool,
    pub router_present: bool,
    pub database_present: bool,
    pub team_present: bool,
    pub user_present: bool,
    pub membership_loaded: bool,
    pub token_org: Option<String>,
    pub team_org: Option<String>,
    pub registry_loaded: bool,
    pub registry: Option<Vec<String>>,
    pub covering: Option<Vec<String>>,
    pub grants: Vec<Vec<String>>,
    pub org_failed: bool,
    pub io_failed: bool,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Complete { groups: Vec<String>, stamp: bool },
    ReadRegistry,
    ReadCovering,
    ReadKey,
    ReadTeam,
    ReadMember { lookup: bool, include: bool },
    ReadProject,
    ReadOrg { id: Option<String> },
}

pub fn decide(facts: &Facts) -> Decision {
    let empty = || Decision::Complete {
        groups: vec![],
        stamp: false,
    };
    if facts.io_failed
        || !facts.model_present
        || !facts.token_present
        || !facts.router_present
        || !facts.database_present
    {
        return empty();
    }
    if !facts.registry_loaded {
        return Decision::ReadRegistry;
    }
    if facts.registry.as_ref().is_some_and(Vec::is_empty) {
        return empty();
    }
    let Some(covering) = &facts.covering else {
        return Decision::ReadCovering;
    };
    let budgeted: BTreeSet<_> = covering
        .iter()
        .filter(|group| {
            facts
                .registry
                .as_ref()
                .is_none_or(|registry| registry.contains(group))
        })
        .collect();
    if budgeted.is_empty() {
        return empty();
    }
    match facts.grants.len() {
        0 => return Decision::ReadKey,
        1 => return Decision::ReadTeam,
        2 => {
            return Decision::ReadMember {
                lookup: facts.team_present && facts.user_present && !facts.membership_loaded,
                include: facts.team_present && facts.user_present,
            };
        }
        3 => return Decision::ReadProject,
        4 => {
            return Decision::ReadOrg {
                id: facts
                    .token_org
                    .as_ref()
                    .filter(|id| !id.is_empty())
                    .or(facts.team_org.as_ref())
                    .cloned(),
            };
        }
        _ => (),
    }
    let granted: BTreeSet<_> = facts
        .grants
        .iter()
        .take(if facts.org_failed { 4 } else { 5 })
        .flatten()
        .collect();
    let groups: Vec<_> = budgeted
        .intersection(&granted)
        .map(|group| (*group).clone())
        .collect();
    Decision::Complete {
        stamp: !groups.is_empty(),
        groups,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::unknown(None, vec!["a", "b"])]
    #[case::known(Some(vec!["a".into()]), vec!["a"])]
    #[case::empty(Some(vec![]), vec![])]
    fn intersects_budgeted_covering_and_granted(
        #[case] registry: Option<Vec<String>>,
        #[case] expected: Vec<&str>,
    ) {
        let facts = Facts {
            model_present: true,
            token_present: true,
            router_present: true,
            database_present: true,
            registry_loaded: true,
            registry,
            covering: Some(vec!["b".into(), "a".into(), "b".into()]),
            grants: vec![
                vec!["a".into()],
                vec![],
                vec!["b".into()],
                vec![],
                vec!["c".into()],
            ],
            ..Default::default()
        };
        assert_eq!(
            decide(&facts),
            Decision::Complete {
                stamp: !expected.is_empty(),
                groups: expected.into_iter().map(str::to_owned).collect()
            }
        );
    }
}
