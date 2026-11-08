use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
pub struct Source {
    pub team_id: Option<String>,
    pub grants: Option<Vec<String>>,
    pub unavailable: bool,
}

#[derive(Deserialize)]
pub struct ConfiguredAgent {
    pub agent_id: String,
    pub groups: Vec<String>,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
pub enum Facts {
    Resolve {
        managed: bool,
        strict: bool,
        role: Option<String>,
        key_team_loaded: bool,
        key_team: Option<Vec<String>>,
        default_deny: Option<bool>,
        caller_loaded: bool,
        own: Option<Vec<String>>,
        ceiling_loaded: bool,
        ceiling: Option<Vec<String>>,
    },
    Failure {
        strict: bool,
        closed: bool,
    },
    OwnerTeam {
        strict: bool,
        permission_id_present: bool,
        permission_present: bool,
    },
    OwnerUnion {
        teams: Vec<Source>,
        withheld: Vec<Source>,
        unreadable: bool,
    },
    BillingTeam {
        sources: Vec<Source>,
        agent_id: String,
    },
    Human {
        sources: Vec<Source>,
        team_id: Option<String>,
    },
    Delegated {
        sources: Vec<Source>,
        agent_id: String,
        team_id: Option<String>,
        explicit: bool,
        allowed: Option<Vec<String>>,
    },
    TeamLookup {
        subject: bool,
        team_id: Option<String>,
        personal: Option<bool>,
        database: bool,
        loaded: bool,
        row_present: bool,
        strict: bool,
    },
    GroupMatches {
        agents: Vec<ConfiguredAgent>,
        groups: Vec<String>,
    },
}

#[derive(Default, Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum Action {
    #[default]
    Complete,
    ReadManaged,
    ReadKeyTeam,
    ReadDefault,
    ReadCaller,
    ReadCeiling,
    ReadPersonalEligibility,
    ReadOwnerTeams,
    ReadTeam,
    ReadDeclared,
    Unavailable,
    SelectTeam,
}

#[derive(Default, Debug, PartialEq, Eq, Serialize)]
pub struct Decision {
    pub action: Action,
    pub grants: Option<Vec<String>>,
    pub team_id: Option<String>,
}

fn action(action: Action) -> Decision {
    Decision {
        action,
        ..Decision::default()
    }
}
fn grants(grants: Option<Vec<String>>) -> Decision {
    Decision {
        grants,
        ..Decision::default()
    }
}
fn intersection(left: Option<Vec<String>>, right: Option<Vec<String>>) -> Option<Vec<String>> {
    match (left, right) {
        (None, right) => right,
        (left, None) => left,
        (Some(left), Some(right)) => {
            Some(left.into_iter().filter(|id| right.contains(id)).collect())
        }
    }
}

pub fn decide(facts: Facts) -> Decision {
    match facts {
        Facts::Resolve {
            managed,
            strict,
            role,
            key_team_loaded,
            key_team,
            default_deny,
            caller_loaded,
            own,
            ceiling_loaded,
            ceiling,
        } => {
            if managed {
                return action(Action::ReadManaged);
            }
            if !key_team_loaded {
                return action(Action::ReadKeyTeam);
            }
            let admin = role.as_deref() == Some("proxy_admin");
            if !strict && !admin && default_deny.is_none() {
                return action(Action::ReadDefault);
            }
            if (strict || (!admin && default_deny == Some(true))) && key_team.is_none() {
                return grants(Some(vec![]));
            }
            if !caller_loaded {
                return action(Action::ReadCaller);
            }
            if !ceiling_loaded {
                return action(Action::ReadCeiling);
            }
            grants(intersection(own, ceiling))
        }
        Facts::Failure { strict, closed } => {
            if strict {
                action(Action::Unavailable)
            } else {
                grants(closed.then(Vec::new))
            }
        }
        Facts::OwnerTeam {
            strict,
            permission_id_present,
            permission_present,
        } => {
            if permission_id_present && !permission_present {
                if strict {
                    action(Action::Unavailable)
                } else {
                    grants(Some(vec![]))
                }
            } else {
                action(Action::ReadDeclared)
            }
        }
        Facts::OwnerUnion {
            teams,
            withheld,
            unreadable,
        } => {
            if teams.iter().any(|source| source.unavailable) {
                return action(Action::Unavailable);
            }
            let restricted = teams.is_empty()
                || unreadable
                || teams.iter().any(|source| source.grants.is_some())
                || withheld
                    .iter()
                    .any(|source| source.unavailable || source.grants.is_some());
            grants(restricted.then(|| {
                teams
                    .into_iter()
                    .flat_map(|source| source.grants.unwrap_or_default())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect()
            }))
        }
        Facts::BillingTeam { sources, agent_id } => {
            let selected = sources
                .iter()
                .filter(|source| {
                    source
                        .grants
                        .as_ref()
                        .is_some_and(|ids| ids.contains(&agent_id))
                })
                .filter_map(|source| source.team_id.clone())
                .min();
            let team_id = selected.or_else(|| {
                sources
                    .iter()
                    .all(|source| source.grants.is_none() && !source.unavailable)
                    .then(|| {
                        sources
                            .iter()
                            .filter_map(|source| source.team_id.clone())
                            .min()
                    })
                    .flatten()
            });
            Decision {
                team_id,
                ..Decision::default()
            }
        }
        Facts::Human { sources, team_id } => grants(Some(
            sources
                .into_iter()
                .filter(|source| source.team_id.is_none() || source.team_id == team_id)
                .flat_map(|source| source.grants.unwrap_or_default())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
        )),
        Facts::Delegated {
            sources,
            agent_id,
            team_id,
            explicit,
            allowed,
        } => {
            if sources.iter().any(|source| {
                source.team_id.is_none()
                    && source
                        .grants
                        .as_ref()
                        .is_some_and(|ids| ids.contains(&agent_id))
            }) {
                return Decision {
                    team_id,
                    ..Decision::default()
                };
            }
            let teams: BTreeSet<_> = sources
                .iter()
                .filter(|source| {
                    source
                        .grants
                        .as_ref()
                        .is_some_and(|ids| ids.contains(&agent_id))
                })
                .filter_map(|source| source.team_id.as_ref())
                .filter(|team| {
                    allowed
                        .as_ref()
                        .is_none_or(|allowed| allowed.contains(team))
                })
                .cloned()
                .collect();
            if team_id.as_ref().is_some_and(|team| teams.contains(team)) {
                return Decision {
                    team_id,
                    ..Decision::default()
                };
            }
            if !explicit && let Some(first) = teams.first() {
                return Decision {
                    team_id: Some(first.clone()),
                    ..Decision::default()
                };
            }
            action(Action::SelectTeam)
        }
        Facts::TeamLookup {
            subject,
            team_id,
            personal,
            database,
            loaded,
            row_present,
            strict,
        } => {
            if !subject {
                return grants(None);
            }
            if team_id.is_none() {
                return match personal {
                    None => action(Action::ReadPersonalEligibility),
                    Some(true) => action(Action::ReadOwnerTeams),
                    Some(false) => grants(None),
                };
            }
            if !database || (loaded && !row_present) {
                return grants(strict.then(Vec::new));
            }
            if !loaded {
                action(Action::ReadTeam)
            } else {
                action(Action::ReadDeclared)
            }
        }
        Facts::GroupMatches { agents, groups } => grants(Some(
            agents
                .into_iter()
                .filter(|agent| agent.groups.iter().any(|id| groups.contains(id)))
                .map(|agent| agent.agent_id)
                .collect(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::legacy(false, false, Action::Complete, None)]
    #[case::closed(false, true, Action::Complete, Some(vec![]))]
    #[case::strict(true, false, Action::Unavailable, None)]
    #[case::strict_closed(true, true, Action::Unavailable, None)]
    fn lookup_failure(
        #[case] strict: bool,
        #[case] closed: bool,
        #[case] expected: Action,
        #[case] expected_grants: Option<Vec<String>>,
    ) {
        assert_eq!(
            decide(Facts::Failure { strict, closed }),
            Decision {
                action: expected,
                grants: expected_grants,
                team_id: None
            }
        );
    }

    #[rstest]
    #[case::unrestricted(None, false, None)]
    #[case::withheld(Some(vec!["a".into()]), false, Some(vec![]))]
    #[case::unreadable(None, true, Some(vec![]))]
    fn withheld_never_widens(
        #[case] withheld: Option<Vec<String>>,
        #[case] unreadable: bool,
        #[case] expected: Option<Vec<String>>,
    ) {
        assert_eq!(
            decide(Facts::OwnerUnion {
                teams: vec![Source {
                    team_id: Some("team".into()),
                    grants: None,
                    unavailable: false
                }],
                withheld: vec![Source {
                    team_id: Some("held".into()),
                    grants: withheld,
                    unavailable: false
                }],
                unreadable
            })
            .grants,
            expected
        );
    }

    #[rstest]
    #[case::implicit(false, Some("a".into()), Action::Complete)]
    #[case::explicit(true, None, Action::SelectTeam)]
    fn delegated_requires_explicit_grant(
        #[case] explicit: bool,
        #[case] team_id: Option<String>,
        #[case] expected: Action,
    ) {
        assert_eq!(
            decide(Facts::Delegated {
                sources: vec![Source {
                    team_id: Some("a".into()),
                    grants: Some(vec!["agent".into()]),
                    unavailable: false
                }],
                agent_id: "agent".into(),
                team_id: Some("other".into()),
                explicit,
                allowed: None
            }),
            Decision {
                action: expected,
                team_id,
                grants: None
            }
        );
    }
}
