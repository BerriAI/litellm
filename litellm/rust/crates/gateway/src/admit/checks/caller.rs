use axum::http::request::Parts;
use litellm_config::GeneralSettings;
use litellm_gateway_auth_core::{Credential, Holder, Named, Team, Via, record::ProjectRecord};

use super::{Admit, Authentication, Reason, Refused, head::TEAM_HEADER};
use crate::{
    callbacks::budgets,
    host::{
        NativeRows,
        auth::{unserved, unserved_membership, unserved_org, unserved_project, unserved_user},
    },
};

/// Each caller the core hands back is matched by name, with no catch-all, so a caller the native
/// host learns to serve does not compile here until it is named native or not. A credential the
/// core did not authenticate is over the error's name, including when a JWT decision needs the
/// model before the body is read. A key that holds a team, owner, organization or project stays
/// with Python over the first unserved column or association. This also applies to mapped keys.
/// Python resolves a JWT customer claim after auth and applies its budgets and models. A key
/// that sends the team header also stays with Python, which reads that header after auth.
pub(super) fn caller(
    admit: &Admit,
    request: &Parts,
    authentication: &Authentication,
) -> Option<Reason> {
    let team_header = || {
        request
            .headers
            .contains_key(TEAM_HEADER)
            .then(|| Reason::new(Refused::Headers, TEAM_HEADER))
    };
    let scope_mappings = || {
        scope_mappings_enforced(&admit.config.general_settings)
            .then(|| Reason::new(Refused::Settings, "enforce_scope_based_access"))
    };
    match authentication {
        Ok(caller) => {
            let key_team = || {
                let project = caller.affiliation.project.as_ref();
                charged(&caller.affiliation.holder)
                    .or_else(|| {
                        caller.affiliation.org.as_ref().map(|org| {
                            org.stored()
                                .and_then(unserved_org)
                                .unwrap_or("organization")
                        })
                    })
                    .or_else(|| {
                        project.map(|project| unserved_project_column(project).unwrap_or("project"))
                    })
                    .map(|over| Reason::new(Refused::Credential, over))
            };
            match &caller.credential {
                Credential::MasterKey => team_header(),
                Credential::Key {
                    via: Via::OwnToken, ..
                } => team_header().or_else(key_team),
                Credential::Key {
                    via: Via::MappedJwt(_),
                    ..
                } => team_header()
                    .or_else(key_team)
                    .or_else(scope_mappings)
                    .or_else(|| customer_claims(admit)),
                Credential::Jwt { .. } => customer_claims(admit),
            }
        }
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
        .or_else(|| holder.membership().and_then(unserved_membership))
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

fn scope_mappings_enforced(general: &GeneralSettings) -> bool {
    general.litellm_jwtauth.as_ref().is_some_and(|jwt| {
        jwt.enforce_scope_based_access
            && jwt
                .scope_mappings
                .as_deref()
                .is_some_and(|mappings| !mappings.is_empty())
    })
}

fn customer_claims(admit: &Admit) -> Option<Reason> {
    maps_customer_claims(&admit.config.general_settings)
        .then(|| Reason::new(Refused::Settings, "customer_jwt_fields"))
}

/// `_maps_customer_claims`, over the settings and every issuer.
fn maps_customer_claims(general: &GeneralSettings) -> bool {
    general.litellm_jwtauth.as_ref().is_some_and(|jwt| {
        let by_issuer = jwt
            .issuers
            .as_deref()
            .unwrap_or_default()
            .iter()
            .flat_map(|issuer| {
                [
                    &issuer.customer_org_alias_jwt_field,
                    &issuer.customer_member_id_jwt_field,
                    &issuer.customer_team_id_jwt_field,
                ]
            });
        [
            &jwt.customer_org_alias_jwt_field,
            &jwt.customer_member_id_jwt_field,
            &jwt.customer_team_id_jwt_field,
        ]
        .into_iter()
        .chain(by_issuer)
        .any(Option::is_some)
    })
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
                via: Via::OwnToken,
            },
            affiliation: Affiliation::nobody(),
            end_user: None,
        })
    }

    fn master_for_a_budgeted_end_user() -> Authenticated<NativeRows> {
        Authenticated::from(Caller::master_key(Some(EndUserRecord {
            end_user_id: Some("e1".to_owned()),
            found: true,
            budget: budget(5.0),
            ..EndUserRecord::default()
        })))
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

    fn jwtauth(settings: &str) -> GeneralSettings {
        let yaml = format!("general_settings:\n  litellm_jwtauth:\n{settings}");
        Config::from_yaml(&yaml).unwrap().general_settings
    }

    #[rstest]
    #[case::no_customer_claim("    team_id_jwt_field: team\n", false)]
    #[case::an_org_alias_in_the_settings("    customer_org_alias_jwt_field: customer.org\n", true)]
    #[case::a_member_id_in_the_settings(
        "    customer_member_id_jwt_field: customer.member\n",
        true
    )]
    #[case::a_team_id_in_the_settings("    customer_team_id_jwt_field: customer.team\n", true)]
    #[case::an_org_alias_in_an_issuer(
        "    issuers:\n      - issuer: https://idp.example\n        customer_org_alias_jwt_field: customer.org\n",
        true
    )]
    #[case::a_member_id_in_an_issuer(
        "    issuers:\n      - issuer: https://idp.example\n        customer_member_id_jwt_field: customer.member\n",
        true
    )]
    #[case::a_team_id_in_an_issuer(
        "    issuers:\n      - issuer: https://idp.example\n        customer_team_id_jwt_field: customer.team\n",
        true
    )]
    fn a_config_that_maps_a_customer_claim_is_one_python_resolves(
        #[case] settings: &str,
        #[case] maps: bool,
    ) {
        assert_eq!(maps_customer_claims(&jwtauth(settings)), maps);
    }

    #[rstest]
    #[case::enforced_with_a_mapping(
        "    enforce_scope_based_access: true\n    scope_mappings:\n      - scope: read\n        models: [claude]\n",
        true
    )]
    #[case::a_mapping_that_is_not_enforced(
        "    scope_mappings:\n      - scope: read\n        models: [claude]\n",
        false
    )]
    #[case::enforced_with_no_mappings("    enforce_scope_based_access: true\n", false)]
    #[case::enforced_with_an_empty_list(
        "    enforce_scope_based_access: true\n    scope_mappings: []\n",
        false
    )]
    fn scope_mappings_gate_a_jwt_only_when_enforced_and_listed(
        #[case] settings: &str,
        #[case] enforced: bool,
    ) {
        assert_eq!(scope_mappings_enforced(&jwtauth(settings)), enforced);
    }
}
