use crate::{Error, protocol::*};

fn allowed() -> Result<Step, Error> {
    Step::allow(Grant::WithinModelBudgets)
}

pub fn check_model_budgets() -> Result<Step, Error> {
    key_budgets(Box::new(allowed))
}

pub fn check_user_model_budgets() -> Result<Step, Error> {
    user_budgets(Box::new(allowed))
}

pub(crate) fn key_budgets(then: Then) -> Result<Step, Error> {
    Step::effect(
        Effect::ReadModelBudgetScope,
        move |scope: ModelBudgetScope| {
            if !scope.key_has_model_budgets || scope.models.is_empty() {
                return user_models(scope.user_has_model_budgets, scope.models, then);
            }
            key_models(
                scope.key_budget_fallbacks,
                scope.models,
                scope.rewritable_model,
                Box::new(move || user_budgets(then)),
            )
        },
    )
}

fn user_budgets(then: Then) -> Result<Step, Error> {
    Step::effect(
        Effect::ReadModelBudgetScope,
        move |scope: ModelBudgetScope| {
            user_models(scope.user_has_model_budgets, scope.models, then)
        },
    )
}

fn key_models(
    fallbacks: std::collections::BTreeMap<String, Vec<String>>,
    models: Vec<String>,
    rewritable: Option<String>,
    then: Then,
) -> Result<Step, Error> {
    let Some((model, rest)) = models.split_first() else {
        return then();
    };
    let model = model.clone();
    let rest = rest.to_vec();
    Step::effect(
        Effect::CheckKeyModelBudget {
            model: model.clone(),
        },
        move |reading: ModelBudgetRead| {
            let exceeded = reading.exceeded();
            let Some(exceeded) = exceeded else {
                return key_models(fallbacks, rest, rewritable, then);
            };
            if Some(&model) != rewritable.as_ref() {
                return Step::deny(exceeded);
            }
            let candidates = fallbacks.get(&model).cloned().unwrap_or_default();
            first_affordable(
                candidates,
                exceeded,
                Box::new(move |fallback| key_models(fallbacks, rest, Some(fallback), then)),
            )
        },
    )
}

type Found = Box<dyn FnOnce(String) -> Result<Step, Error> + Send + Sync>;

fn first_affordable(
    candidates: Vec<String>,
    exceeded: ModelBudgetExceeded,
    found: Found,
) -> Result<Step, Error> {
    let Some((candidate, rest)) = candidates.split_first() else {
        return Step::deny(exceeded);
    };
    let candidate = candidate.clone();
    let rest = rest.to_vec();
    Step::effect(
        Effect::CheckKeyModelBudget {
            model: candidate.clone(),
        },
        move |reading: ModelBudgetRead| {
            let over = reading.exceeded();
            if over.is_some() {
                return first_affordable(rest, exceeded, found);
            }
            Step::effect(
                Effect::AuthorizeBudgetFallback {
                    model: candidate.clone(),
                },
                move |facts: crate::fallback_access::Facts| {
                    match crate::fallback_access::decide(facts)? {
                        crate::fallback_access::Decision::Allow => (),
                        crate::fallback_access::Decision::Deny {
                            denial: Some(denial @ crate::model_access::Decision::Depth { .. }),
                        } => return Step::deny(Denial::ModelAccessDenied { decision: denial }),
                        crate::fallback_access::Decision::Deny { .. } => {
                            return Step::deny(exceeded);
                        }
                        _ => return Err(Error::InvalidAnswer),
                    }
                    Step::effect(
                        Effect::RewriteRequestModel {
                            model: candidate.clone(),
                        },
                        move |(): ()| found(candidate),
                    )
                },
            )
        },
    )
}

fn user_models(enabled: bool, models: Vec<String>, then: Then) -> Result<Step, Error> {
    if !enabled {
        return then();
    }
    let Some((model, rest)) = models.split_first() else {
        return then();
    };
    let rest = rest.to_vec();
    Step::effect(
        Effect::CheckUserModelBudget {
            model: model.clone(),
        },
        move |reading: ModelBudgetRead| match reading.exceeded() {
            Some(exceeded) => Step::deny(exceeded),
            None => user_models(enabled, rest, then),
        },
    )
}

pub fn check_team_member_model_budgets() -> Result<Step, Error> {
    Step::effect(
        Effect::ReadTeamMemberModelBudgetScope,
        |scope: TeamMemberModelBudgetScope| {
            if !scope.has_model_budgets {
                return allowed();
            }
            member_models(scope.models, scope.rewritable_model, scope.can_fall_back)
        },
    )
}

fn member_models(
    models: Vec<String>,
    rewritable: Option<String>,
    can_fall_back: bool,
) -> Result<Step, Error> {
    let Some((model, rest)) = models.split_first() else {
        return allowed();
    };
    let model = model.clone();
    let rest = rest.to_vec();
    Step::effect(
        Effect::CheckTeamMemberModelBudget {
            model: model.clone(),
        },
        move |reading: ModelBudgetRead| {
            let exceeded = reading.exceeded();
            let Some(exceeded) = exceeded else {
                return member_models(rest, rewritable, can_fall_back);
            };
            if Some(&model) != rewritable.as_ref() || !can_fall_back {
                return Step::deny(exceeded);
            }
            Step::effect(
                Effect::ReadTeamMemberFallbacks {
                    model: model.clone(),
                },
                move |targets: Vec<FallbackTarget>| {
                    first_usable(
                        targets
                            .into_iter()
                            .filter(|target| target.model != model)
                            .collect(),
                        exceeded,
                        Box::new(move |fallback| {
                            member_models(rest, Some(fallback), can_fall_back)
                        }),
                    )
                },
            )
        },
    )
}

fn first_usable(
    targets: Vec<FallbackTarget>,
    exceeded: ModelBudgetExceeded,
    found: Found,
) -> Result<Step, Error> {
    let Some((target, rest)) = targets.split_first() else {
        return Step::deny(exceeded);
    };
    let target = target.clone();
    let rest = rest.to_vec();
    Step::effect(
        Effect::VetTeamMemberFallback {
            target: target.clone(),
        },
        move |facts: crate::fallback_candidate::Facts| {
            let crate::fallback_candidate::Decision::Complete { usable, .. } =
                crate::fallback_candidate::decide(&facts)?
            else {
                return Err(Error::InvalidAnswer);
            };
            if !usable {
                return first_usable(rest, exceeded, found);
            }
            Step::effect(
                Effect::RerouteTeamMemberRequest {
                    model: target.model.clone(),
                },
                move |(): ()| found(target.model),
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn unauthorized_affordable_fallback_denies_without_trying_the_next_one() {
        let exceeded = ModelBudgetExceeded {
            message: "over budget".into(),
            current_cost: 2.0,
            max_budget: 1.0,
            entity_type: None,
            entity_id: None,
        };
        let read = first_affordable(
            vec!["first".into(), "second".into()],
            exceeded,
            Box::new(|_| allowed()),
        )
        .unwrap();
        assert!(
            matches!(read.node, Node::Effect(Effect::CheckKeyModelBudget { ref model }) if model == "first")
        );
        let authorize = (read.resume.unwrap())(Answer::ModelBudgetData(ModelBudgetRead {
            snapshot: crate::model_budget_data::Snapshot {
                max_budget: None,
                local_spend: None,
                legacy_spend: None,
                shared_spend: None,
                shared_started_at: None,
                shared_window_seconds: None,
                now: 0.0,
            },
            message: String::new(),
            entity_type: None,
            entity_id: None,
        }))
        .unwrap();
        assert!(
            matches!(authorize.node, Node::Effect(Effect::AuthorizeBudgetFallback { ref model }) if model == "first")
        );
        let result = (authorize.resume.unwrap())(Answer::FallbackAccess(Box::new(
            crate::fallback_access::Facts {
                refuse_agent_keys: true,
                agent_id: Some("agent".into()),
                key: None,
                team: None,
                embedded_team: None,
                assigned_models: None,
                member: None,
                project: None,
                team_present: false,
                embedded_team_grants: vec![],
                member_grants: vec![],
                project_grants: vec![],
            },
        )))
        .unwrap();
        assert!(matches!(
            result.node,
            Node::Outcome(Outcome::Deny {
                denial: Denial::ModelBudgetExceeded { .. }
            })
        ));
        assert!(result.resume.is_none());
    }

    #[rstest]
    fn user_entry_never_checks_key_budgets() {
        let step = check_user_model_budgets().unwrap();
        let result = (step.resume.unwrap())(Answer::ModelScope(ModelBudgetScope {
            models: vec!["model".into()],
            rewritable_model: Some("model".into()),
            key_has_model_budgets: true,
            key_budget_fallbacks: Default::default(),
            user_has_model_budgets: true,
        }))
        .unwrap();
        assert!(
            matches!(result.node, Node::Effect(Effect::CheckUserModelBudget { ref model }) if model == "model")
        );
    }
}
