use litellm_gateway_auth_core::Credential;

use super::{Admit, Authentication, Reason, Refused};
use crate::callbacks::budgets;

/// Each caller the core hands back is matched by name, with no catch-all, so a caller the native
/// host learns to serve does not compile here until it is named native or not. A credential the
/// core did not authenticate is over the error's name.
pub(super) fn caller(_: &Admit, authentication: &Authentication) -> Option<Reason> {
    match authentication {
        Ok(caller) => match &caller.credential {
            Credential::MasterKey | Credential::Key { .. } => None,
        },
        Err(error) => Some(Reason::new(
            Refused::Credential,
            <&'static str>::from(error),
        )),
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
}
