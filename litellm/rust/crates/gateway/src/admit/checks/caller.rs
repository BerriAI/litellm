use litellm_gateway_auth_core::{Credential, Holder, Named, Team, record::ProjectRecord};

use super::{Admit, Authentication, Reason, Refused};
use crate::{
    callbacks::budgets,
    host::{
        NativeRows,
        auth::{unserved, unserved_project, unserved_user},
    },
};

/// Each caller the core hands back is matched by name, with no catch-all, so a caller the native
/// host learns to serve does not compile here until it is named native or not. A credential the
/// core did not authenticate is over the error's name. A key in a team or with an owner stays
/// with Python: over the first column its team, then its owner, holds that nothing native reads
/// yet, and otherwise over `team` or `user`, since the native spend log charges neither yet. A key
/// in a project stays with Python the same way, over `project` once neither of those keeps it
/// there.
pub(super) fn caller(_: &Admit, authentication: &Authentication) -> Option<Reason> {
    match authentication {
        Ok(caller) => match &caller.credential {
            Credential::MasterKey => None,
            Credential::Key { .. } => {
                let project = caller.affiliation.project.as_ref();
                charged(&caller.affiliation.holder)
                    .or_else(|| {
                        project.map(|project| unserved_project_column(project).unwrap_or("project"))
                    })
                    .map(|over| Reason::new(Refused::Credential, over))
            }
        },
        Err(error) => Some(Reason::new(
            Refused::Credential,
            <&'static str>::from(error),
        )),
    }
}

/// What keeps a key's holder with Python, if it holds a team or an owner.
fn charged(holder: &Holder<NativeRows>) -> Option<&'static str> {
    let team = holder.team();
    let user = holder.user();
    team.and_then(unserved_team_column)
        .or_else(|| user.and_then(Named::stored).and_then(unserved_user))
        .or_else(|| team.map(|_| "team"))
        .or_else(|| user.map(|_| "user"))
}

/// The column of the caller's team that keeps the caller with Python. A team no row stands behind
/// holds none. A key's own copy of its team is no row the native app reads, so it is never served.
fn unserved_team_column(team: &Team<NativeRows>) -> Option<&'static str> {
    match team {
        Team::Row(record) => unserved(record),
        Team::Unstored(_) => None,
        Team::KeyCopy(_) => Some("team"),
    }
}

/// The column of the caller's project that keeps the caller with Python. A project no row stands
/// behind holds none.
fn unserved_project_column(project: &Named<ProjectRecord>) -> Option<&'static str> {
    match project {
        Named::Stored(record) => unserved_project(record),
        Named::Unstored(_) => None,
    }
}

/// Python reserves what a call may cost against every budget its caller holds before the call
/// runs, unless `general_settings.disable_budget_reservation` turns that off. The gateway reserves
/// nothing, so a caller with a budget goes to Python while reservation is on.
pub(super) fn budgets(admit: &Admit, authentication: &Authentication) -> Option<Reason> {
    let caller = authentication.as_ref().ok()?;
    let reserves = !admit.config.general_settings.disable_budget_reservation
        && !budgets::counters(caller, budgets::Settings::default()).is_empty();
    reserves.then(|| {
        Reason::new(
            Refused::Settings,
            "general_settings.disable_budget_reservation",
        )
    })
}

#[cfg(test)]
mod tests {
    use litellm_config::Config;
    use litellm_gateway_auth::Authenticated;
    use litellm_gateway_auth_core::{
        Affiliation, Caller,
        record::{Budget, EndUserRecord},
    };
    use rstest::rstest;

    use super::*;
    use crate::{
        host::{Key, NativeRows},
        test_support::master,
    };

    fn budget(max: f64) -> Budget {
        Budget {
            max: Some(max),
            spend: 0.0,
        }
    }

    fn budgeted_key(max: f64) -> Authenticated<NativeRows> {
        Authenticated::from(Caller {
            credential: Credential::Key {
                key: Key::stored("hash", None).with_budget(budget(max)),
            },
            affiliation: Affiliation::nobody(),
            end_user: EndUserRecord::default(),
        })
    }

    fn master_for_a_budgeted_end_user() -> Authenticated<NativeRows> {
        Authenticated::from(Caller::master_key(EndUserRecord {
            end_user_id: Some("e1".to_owned()),
            found: true,
            budget: budget(5.0),
            ..EndUserRecord::default()
        }))
    }

    fn reserving(disabled: bool) -> Admit {
        let config = Config::from_yaml(&format!(
            "general_settings:\n  disable_budget_reservation: {disabled}\n"
        ))
        .unwrap();
        Admit::over(&config, Vec::new())
    }

    #[rstest]
    #[case::a_budgeted_key_while_python_reserves(budgeted_key(10.0), false, true)]
    #[case::a_key_at_a_budget_of_zero(budgeted_key(0.0), false, true)]
    #[case::the_master_key_for_a_budgeted_end_user(master_for_a_budgeted_end_user(), false, true)]
    #[case::a_budgeted_key_once_reservation_is_off(budgeted_key(10.0), true, false)]
    #[case::a_caller_with_no_budget(master(), false, false)]
    fn a_caller_python_reserves_for_goes_to_python(
        #[case] caller: Authenticated<NativeRows>,
        #[case] disabled: bool,
        #[case] to_python: bool,
    ) {
        let reason = budgets(&reserving(disabled), &Ok(caller));

        assert_eq!(
            reason,
            to_python.then(|| Reason::new(
                Refused::Settings,
                "general_settings.disable_budget_reservation"
            ))
        );
    }

    fn key_of(holder: Holder<NativeRows>, project: Option<&str>) -> Authenticated<NativeRows> {
        Authenticated::from(Caller {
            credential: Credential::Key {
                key: Key::stored("hash", None),
            },
            affiliation: Affiliation {
                holder,
                org: None,
                project: project.map(|id| Named::Unstored(id.to_owned())),
            },
            end_user: EndUserRecord::default(),
        })
    }

    /// The standard logging payload of a native call logs a team's, owner's and project's id and
    /// alias but none of their spend, budget or email, so a key that names any of them must stay
    /// with Python until the builder logs those too.
    #[rstest]
    #[case::a_key_in_a_team(key_of(Holder::Team(Team::Unstored("t1".to_owned())), None), "team")]
    #[case::a_key_with_an_owner(key_of(Holder::User(Named::Unstored("u1".to_owned())), None), "user")]
    #[case::a_key_in_a_project(key_of(Holder::Nobody, Some("p1")), "project")]
    fn a_key_in_a_team_with_an_owner_or_in_a_project_goes_to_python(
        #[case] caller: Authenticated<NativeRows>,
        #[case] over: &'static str,
    ) {
        assert_eq!(
            super::caller(&reserving(false), &Ok(caller)),
            Some(Reason::new(Refused::Credential, over))
        );
    }
}
