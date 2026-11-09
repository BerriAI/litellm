use litellm_secret_value::SecretValue;
use sha2::{Digest, Sha256};

use crate::{Error, model_budgets, protocol::*};

struct KeyCheck {
    settings: Settings,
    now: DateTime,
    is_llm_api_route: bool,
    key_label: SecretValue,
}

pub fn start(facts: RequestFacts, settings: Settings, now: DateTime) -> Result<Step, Error> {
    let Some(master_key) = &settings.master_key else {
        return Step::allow(Grant::NoMasterKey {
            token: facts.token.map(|token| token.expose().to_owned()),
        });
    };
    let Some(token) = facts.token.clone() else {
        return Step::deny(Denial::MissingKey);
    };
    if token.expose().is_empty() {
        return Step::deny(Denial::MalformedKey);
    }
    if facts.route == "/user/auth" {
        return if settings.allow_user_auth {
            Step::allow(Grant::UserAuthRoute)
        } else {
            Step::deny(Denial::UserAuthDisabled)
        };
    }
    if &token == master_key {
        return Step::effect(
            Effect::ResolveEndUser {
                prefetch_key: false,
            },
            |(): ()| Step::allow(Grant::MasterKey),
        );
    }
    let hashed_token = format!("{:x}", Sha256::digest(token.expose().as_bytes()));
    Step::effect(
        Effect::ResolveEndUser { prefetch_key: true },
        move |(): ()| {
            Step::effect(
                Effect::ResolvePresentedKey {
                    hashed_token: hashed_token.clone(),
                },
                move |presented: Option<PresentedKey>| {
                    after_presented_key(presented, token, hashed_token, facts, settings, now)
                },
            )
        },
    )
}

pub fn start_resolved_key(
    facts: RequestFacts,
    settings: Settings,
    now: DateTime,
) -> Result<Step, Error> {
    check_key(KeyCheck {
        settings,
        now,
        is_llm_api_route: facts.is_llm_api_route,
        key_label: facts.token.unwrap_or_else(|| "".into()),
    })
}

fn after_presented_key(
    presented: Option<PresentedKey>,
    token: SecretValue,
    hashed_token: String,
    facts: RequestFacts,
    settings: Settings,
    now: DateTime,
) -> Result<Step, Error> {
    if let Some(PresentedKey::AdminKey { expires }) = &presented {
        if let Some(expires) = expires
            && expires.before(&now)?
        {
            let expires = expires.clone();
            return Step::effect(Effect::EvictPresentedKey { hashed_token }, move |(): ()| {
                Step::deny(Denial::ExpiredKey {
                    expires,
                    now,
                    key_label: token.expose().to_owned(),
                })
            });
        }
        return Step::allow(Grant::CachedAdminKey);
    }
    if settings.master_key_only_routes.contains(&facts.route) {
        return Step::deny(Denial::MasterKeyOnlyRoute);
    }
    if !settings.has_db {
        return Step::deny(Denial::NoDatabase);
    }
    if presented.is_some() {
        let check = KeyCheck {
            settings,
            now,
            is_llm_api_route: facts.is_llm_api_route,
            key_label: token,
        };
        return Step::effect(
            Effect::PrepareKey {
                loaded_from_db: false,
            },
            move |(): ()| check_key(check),
        );
    }
    if !token.expose().starts_with("sk-") {
        return Step::deny(Denial::NotAVirtualKey);
    }
    let check = KeyCheck {
        settings,
        now,
        is_llm_api_route: facts.is_llm_api_route,
        key_label: hashed_token.clone().into(),
    };
    Step::effect(Effect::LoadKey { hashed_token }, move |(): ()| {
        Step::effect(
            Effect::PrepareKey {
                loaded_from_db: true,
            },
            move |(): ()| check_key(check),
        )
    })
}

fn check_key(check: KeyCheck) -> Result<Step, Error> {
    Step::effect(Effect::InspectKey, move |key: KeyFacts| {
        if key.blocked {
            return Step::deny(Denial::KeyBlocked);
        }
        Step::effect(
            Effect::EnforceModelAccess,
            move |facts: crate::key_model_access::Facts| match crate::key_model_access::decide(
                facts,
            )? {
                crate::key_model_access::Decision::Allow => {
                    Step::effect(Effect::PrefetchAuthObjects, move |(): ()| {
                        check_owner(check, key)
                    })
                }
                crate::key_model_access::Decision::Deny { denial } => {
                    Step::deny(Denial::ModelAccessDenied { decision: denial })
                }
                _ => Err(Error::InvalidAnswer),
            },
        )
    })
}

fn check_owner(check: KeyCheck, key: KeyFacts) -> Result<Step, Error> {
    let Some(user_id) = key.user_id.clone() else {
        return check_budget_exemption(check, key);
    };
    Step::effect(
        Effect::LoadKeyOwner {
            user_id: user_id.clone(),
        },
        move |owner: Option<OwnerFacts>| {
            if owner.is_some_and(|owner| owner.deactivated) {
                return Step::deny(Denial::OwnerDeactivated { user_id });
            }
            check_budget_exemption(check, key)
        },
    )
}

fn check_budget_exemption(check: KeyCheck, key: KeyFacts) -> Result<Step, Error> {
    Step::effect(
        Effect::CheckBudgetExemption,
        move |facts: crate::budget_pricing::ExemptionFacts| {
            let decision = crate::budget_pricing::exemption(&facts);
            if decision.read_pass_through
                || decision.read_price_cache
                || decision.read_model.is_some()
            {
                return Err(Error::InvalidAnswer);
            }
            let exempt = decision.exempt;
            if exempt || key.team_member_spend.is_none() || !check.settings.has_db {
                return check_expiry(check, key, exempt);
            }
            let (Some(user_id), Some(team_id)) = (key.user_id.clone(), key.team_id.clone()) else {
                return check_expiry(check, key, exempt);
            };
            Step::effect(
                Effect::ReadTeamMemberBudget {
                    user_id: user_id.clone(),
                    team_id: team_id.clone(),
                },
                move |budget: Option<TeamMemberBudget>| {
                    let Some(budget) = budget else {
                        return check_expiry(check, key, false);
                    };
                    if budget.spend < budget.max_budget {
                        return check_expiry(check, key, false);
                    }
                    Step::effect(
                        Effect::AlertTeamMemberBudget {
                            user_id: user_id.clone(),
                            team_id: team_id.clone(),
                            budget: budget.clone(),
                        },
                        move |(): ()| {
                            Step::deny(Denial::TeamMemberBudgetExceeded {
                                user_id,
                                team_id,
                                spend: budget.spend,
                                max_budget: budget.max_budget,
                            })
                        },
                    )
                },
            )
        },
    )
}

fn check_expiry(check: KeyCheck, key: KeyFacts, exempt: bool) -> Result<Step, Error> {
    if let Some(expires) = &key.expires
        && expires.before(&check.now)?
    {
        return Step::deny(Denial::ExpiredKey {
            expires: expires.clone(),
            now: check.now,
            key_label: check.key_label.expose().to_owned(),
        });
    }
    if exempt {
        return load_context(check, key, None);
    }
    Step::effect(Effect::SendKeyBudgetAlerts, move |(): ()| {
        check_key_budget(check, key)
    })
}

fn check_key_budget(check: KeyCheck, key: KeyFacts) -> Result<Step, Error> {
    let Some(max_budget) = key.max_budget.filter(|_| check.is_llm_api_route) else {
        return check_soft_budget(check, key, None);
    };
    Step::effect(Effect::ReadKeySpend { max_budget }, move |spend: f64| {
        Step::effect(Effect::AlertKeyBudget { spend }, move |(): ()| {
            if !max_budget.is_finite() || spend < max_budget {
                return check_soft_budget(check, key, None);
            }
            if let Some(throttle) = key.budget_throttle_pct {
                return check_soft_budget(check, key, Some(throttle));
            }
            Step::deny(Denial::KeyBudgetExceeded { spend, max_budget })
        })
    })
}

fn check_soft_budget(check: KeyCheck, key: KeyFacts, throttle: Option<f64>) -> Result<Step, Error> {
    let alert = key
        .soft_budget
        .is_some_and(|budget| budget != 0.0 && key.spend >= budget);
    let then =
        move || model_budgets::key_budgets(Box::new(move || load_context(check, key, throttle)));
    if alert {
        return Step::effect(Effect::AlertKeySoftBudget, move |(): ()| then());
    }
    then()
}

fn load_context(check: KeyCheck, key: KeyFacts, throttle: Option<f64>) -> Result<Step, Error> {
    Step::effect(Effect::LoadKeyContext, move |(): ()| {
        if !(check.settings.proxy_max_budget > 0.0 && check.settings.has_db) {
            return grant(key, throttle);
        }
        Step::effect(
            Effect::ReadProxySpend,
            move |spend: Option<f64>| match spend {
                None => grant(key, throttle),
                Some(spend) => Step::effect(Effect::AlertProxyBudget { spend }, move |(): ()| {
                    grant(key, throttle)
                }),
            },
        )
    })
}

fn grant(key: KeyFacts, throttle: Option<f64>) -> Result<Step, Error> {
    if !key.has_token {
        return Step::deny(Denial::KeyWithoutToken);
    }
    Step::allow(Grant::VirtualKey {
        budget_throttle_pct: throttle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn check() -> KeyCheck {
        KeyCheck {
            settings: Settings {
                master_key: Some("sk-master".into()),
                allow_user_auth: false,
                has_db: true,
                master_key_only_routes: ["/admin".into()].into(),
                proxy_max_budget: 0.0,
            },
            now: DateTime {
                wall_microseconds: 100,
                offset_microseconds: Some(0),
                timezone_id: 1,
                reference: 0,
            },
            is_llm_api_route: true,
            key_label: "credential-value".into(),
        }
    }

    #[fixture]
    fn key() -> KeyFacts {
        KeyFacts {
            has_token: true,
            blocked: false,
            expires: None,
            spend: 0.0,
            soft_budget: None,
            max_budget: None,
            team_member_spend: None,
            user_id: None,
            team_id: None,
            budget_throttle_pct: None,
        }
    }

    #[rstest]
    fn blocked_key_stops_before_model_access(check: KeyCheck, key: KeyFacts) {
        let step = check_key(check).unwrap();
        assert!(matches!(step.node, Node::Effect(Effect::InspectKey)));
        let result = (step.resume.unwrap())(Answer::Key(KeyFacts {
            blocked: true,
            ..key
        }))
        .unwrap();
        assert!(matches!(
            result.node,
            Node::Outcome(Outcome::Deny {
                denial: Denial::KeyBlocked
            })
        ));
        assert!(result.resume.is_none());
    }

    #[rstest]
    #[case::at_limit(1.0, 1.0, true)]
    #[case::below_limit(0.9, 1.0, false)]
    #[case::nan_limit(1.0, f64::NAN, false)]
    #[case::infinite_limit(1.0, f64::INFINITY, false)]
    #[case::nan_spend(f64::NAN, 1.0, true)]
    fn budget_alert_precedes_the_decision(
        check: KeyCheck,
        key: KeyFacts,
        #[case] spend: f64,
        #[case] max_budget: f64,
        #[case] denied: bool,
    ) {
        let read = check_key_budget(
            check,
            KeyFacts {
                max_budget: Some(max_budget),
                ..key
            },
        )
        .unwrap();
        assert!(matches!(
            read.node,
            Node::Effect(Effect::ReadKeySpend { .. })
        ));
        let alert = (read.resume.unwrap())(Answer::Spend(spend)).unwrap();
        assert!(matches!(
            alert.node,
            Node::Effect(Effect::AlertKeyBudget { .. })
        ));
        let result = (alert.resume.unwrap())(Answer::Unit).unwrap();
        if denied {
            assert!(matches!(
                result.node,
                Node::Outcome(Outcome::Deny {
                    denial: Denial::KeyBudgetExceeded { .. }
                })
            ));
        } else {
            assert!(matches!(
                result.node,
                Node::Effect(Effect::ReadModelBudgetScope)
            ));
        }
    }

    #[rstest]
    fn cached_admin_skips_route_and_database_checks(check: KeyCheck) {
        let settings = Settings {
            has_db: false,
            ..check.settings
        };
        let step = start(
            RequestFacts {
                route: "/admin".into(),
                token: Some("other".into()),
                is_llm_api_route: false,
            },
            settings,
            check.now,
        )
        .unwrap();
        let lookup = (step.resume.unwrap())(Answer::Unit).unwrap();
        let result = (lookup.resume.unwrap())(Answer::Presented(Some(PresentedKey::AdminKey {
            expires: None,
        })))
        .unwrap();
        assert!(matches!(
            result.node,
            Node::Outcome(Outcome::Allow {
                grant: Grant::CachedAdminKey
            })
        ));
    }

    #[rstest]
    fn public_expiry_payload_does_not_leak_into_debug(check: KeyCheck) {
        let denial = Denial::ExpiredKey {
            expires: check.now.clone(),
            now: check.now,
            key_label: check.key_label.expose().to_owned(),
        };
        let debug = format!("{denial:?}");
        assert!(!debug.contains("credential-value"));
        assert!(debug.contains("REDACTED"));
    }
}
