use serde::{Deserialize, Serialize};

use crate::access::{self, AccessFacts, AccessRead};

#[derive(Deserialize)]
#[serde(tag = "kind")]
pub enum Facts {
    Fetch {
        team_id: Option<String>,
        ui_team_id: String,
        user_id: Option<String>,
        project_id: Option<String>,
        end_user_id: Option<String>,
    },
    Team {
        team_id: Option<String>,
        ui_team_id: String,
        failed: bool,
        missing: bool,
        recorded_models: Vec<String>,
        allow_unavailable: Option<bool>,
    },
    Admin {
        role: Option<String>,
        user_id: Option<String>,
        admin_id: String,
        user_row: bool,
    },
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum TeamSource {
    None,
    Token,
    Loaded,
    ReadUnavailablePolicy,
    Failure,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Plan {
    Fetch {
        team: bool,
        user: bool,
        project: bool,
        end_user: bool,
    },
    Team {
        source: TeamSource,
    },
    Admin {
        force_admin: bool,
        preserve_budget: bool,
    },
}

pub fn plan(facts: Facts) -> Plan {
    match facts {
        Facts::Fetch {
            team_id,
            ui_team_id,
            user_id,
            project_id,
            end_user_id,
        } => Plan::Fetch {
            team: team_id.as_ref().is_some_and(|id| id != &ui_team_id),
            user: user_id.is_some(),
            project: project_id.is_some(),
            end_user: end_user_id.is_some_and(|id| !id.is_empty()),
        },
        Facts::Team {
            team_id,
            ui_team_id,
            failed,
            missing,
            recorded_models,
            allow_unavailable,
        } => {
            let dashboard = team_id.as_ref() == Some(&ui_team_id);
            let source = if !failed {
                if dashboard {
                    TeamSource::Token
                } else {
                    TeamSource::Loaded
                }
            } else if team_id.is_none() {
                TeamSource::None
            } else {
                let decision = access::decide(AccessFacts::TeamLookup {
                    dashboard,
                    missing,
                    recorded_models,
                    allow_unavailable,
                });
                if decision.read == Some(AccessRead::UnavailablePolicy) {
                    TeamSource::ReadUnavailablePolicy
                } else if decision.denial.is_some() {
                    TeamSource::Failure
                } else {
                    TeamSource::Token
                }
            };
            Plan::Team { source }
        }
        Facts::Admin {
            role,
            user_id,
            admin_id,
            user_row,
        } => Plan::Admin {
            force_admin: role.as_deref() == Some("proxy_admin"),
            preserve_budget: user_row && user_id.as_ref() != Some(&admin_id),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::absent(None, false)]
    #[case::dashboard(Some("ui"), false)]
    #[case::empty(Some(""), true)]
    #[case::team(Some("team"), true)]
    fn fetch_team(#[case] team: Option<&str>, #[case] expected: bool) {
        assert_eq!(
            plan(Facts::Fetch {
                team_id: team.map(str::to_owned),
                ui_team_id: "ui".into(),
                user_id: Some("".into()),
                project_id: Some("".into()),
                end_user_id: Some("".into())
            }),
            Plan::Fetch {
                team: expected,
                user: true,
                project: true,
                end_user: false
            }
        );
    }

    #[rstest]
    #[case::master(Some("admin"), true, false)]
    #[case::personal(Some("user"), true, true)]
    #[case::no_row(Some("user"), false, false)]
    #[case::missing_id(None, true, true)]
    fn admin_budget_projection(
        #[case] user: Option<&str>,
        #[case] user_row: bool,
        #[case] preserve_budget: bool,
    ) {
        assert_eq!(
            plan(Facts::Admin {
                role: Some("proxy_admin".into()),
                user_id: user.map(str::to_owned),
                admin_id: "admin".into(),
                user_row
            }),
            Plan::Admin {
                force_admin: true,
                preserve_budget
            }
        );
    }
}
