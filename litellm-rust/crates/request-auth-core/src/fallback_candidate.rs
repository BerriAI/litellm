use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use strum::VariantArray;

use crate::{
    Error, budget_pricing, common_budgets, fallback_access, model_access, model_attribution,
    model_budget_data, route_access, unpriced_models,
};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, strum::VariantArray)]
pub enum Entity {
    Key,
    User,
    EndUser,
    Team,
    TeamMember,
}

#[derive(Clone, Copy, Default, Deserialize, PartialEq, Eq, strum::VariantArray)]
pub enum Membership {
    #[default]
    Unread,
    Loaded,
    Failed,
}

#[derive(Clone, Deserialize)]
pub struct BudgetScope {
    pub entity: Entity,
    pub identity_present: bool,
    pub configured: bool,
}

#[derive(Clone, Deserialize)]
pub struct BudgetRead {
    pub model_index: usize,
    pub entity: Entity,
    pub snapshot: model_budget_data::Snapshot,
}

#[derive(Clone, Default, Deserialize)]
pub struct Facts {
    pub router_present: bool,
    pub user_present: bool,
    pub team_present: bool,
    pub membership: Membership,
    pub block_unpriced: bool,
    pub route: Option<route_access::RouteContext>,
    pub unpriced: Option<unpriced_models::Facts>,
    pub customer_present: bool,
    pub customer_grants: Option<Vec<String>>,
    pub customer: Option<model_access::Facts>,
    pub price: Option<budget_pricing::ModelPriceFacts>,
    pub budget_models: Option<Vec<String>>,
    pub budget_scopes: Option<Vec<BudgetScope>>,
    pub budgets: Vec<BudgetRead>,
    pub authorization: Option<fallback_access::Facts>,
    pub attribution_reset: bool,
    pub attribution: Option<model_attribution::Facts>,
    pub attribution_stamped: bool,
    pub group_budgets_read: bool,
    pub group_budgets: Vec<common_budgets::BudgetFacts>,
    pub io_failed: bool,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Complete { usable: bool, log_failure: bool },
    ReadMembership,
    ReadRoute,
    ReadUnpriced,
    ReadCustomerGrants,
    ReadCustomer,
    ReadPrice,
    ReadBudgetModels,
    ReadBudgetScopes,
    ReadBudget { model_index: usize, entity: Entity },
    ReadAuthorization,
    ResetAttribution,
    ReadAttribution,
    StampAttribution { groups: Vec<String> },
    ReadGroupBudgets { groups: Vec<String> },
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    let complete = |usable, log_failure| {
        Ok(Decision::Complete {
            usable,
            log_failure,
        })
    };
    if !facts.router_present
        || !facts.user_present
        || !facts.team_present
        || facts.membership == Membership::Failed
    {
        return complete(false, false);
    }
    if facts.membership == Membership::Unread {
        return Ok(Decision::ReadMembership);
    }
    if facts.io_failed {
        return complete(false, false);
    }
    if facts.block_unpriced {
        let Some(route) = &facts.route else {
            return Ok(Decision::ReadRoute);
        };
        if route_access::decide(route_access::Request::Classify {
            context: route.clone(),
            kind: route_access::Classification::Llm,
        })?
        .allowed
        {
            let Some(unpriced) = &facts.unpriced else {
                return Ok(Decision::ReadUnpriced);
            };
            match unpriced_models::decide(unpriced) {
                unpriced_models::Decision::Complete { unpriced: true } => {
                    return complete(false, false);
                }
                unpriced_models::Decision::Complete { unpriced: false } => (),
                _ => return Err(Error::InvalidAnswer),
            }
        }
    }
    if facts.customer_present {
        let Some(grants) = &facts.customer_grants else {
            return Ok(Decision::ReadCustomerGrants);
        };
        if !grants.is_empty() {
            let Some(customer) = &facts.customer else {
                return Ok(Decision::ReadCustomer);
            };
            match model_access::decide(model_access::Request::Customer(customer.clone()))? {
                model_access::Response::Decision(decision) if decision.allowed() => (),
                model_access::Response::Decision(_) => return complete(false, true),
                _ => return Err(Error::InvalidAnswer),
            }
        }
    }
    let Some(price) = &facts.price else {
        return Ok(Decision::ReadPrice);
    };
    let priced = budget_pricing::price(price);
    if priced.read_deployments {
        return Err(Error::InvalidAnswer);
    }
    if !priced.zero_cost {
        let Some(models) = &facts.budget_models else {
            return Ok(Decision::ReadBudgetModels);
        };
        let Some(scopes) = &facts.budget_scopes else {
            return Ok(Decision::ReadBudgetScopes);
        };
        let mut seen = BTreeSet::new();
        for (model_index, model) in models.iter().enumerate() {
            if !seen.insert(model) {
                continue;
            }
            for entity in Entity::VARIANTS {
                if !scopes.iter().any(|scope| {
                    scope.entity == *entity && scope.identity_present && scope.configured
                }) {
                    continue;
                }
                let Some(reading) = facts
                    .budgets
                    .iter()
                    .find(|read| read.model_index == model_index && read.entity == *entity)
                else {
                    return Ok(Decision::ReadBudget {
                        model_index,
                        entity: *entity,
                    });
                };
                let budget = model_budget_data::check(&reading.snapshot);
                if budget.read.is_some() {
                    return Err(Error::InvalidAnswer);
                }
                if budget.denied {
                    return complete(false, false);
                }
            }
        }
    }
    let Some(authorization) = &facts.authorization else {
        return Ok(Decision::ReadAuthorization);
    };
    match fallback_access::decide(authorization.clone())? {
        fallback_access::Decision::Allow => (),
        fallback_access::Decision::Deny { .. } => return complete(false, false),
        _ => return Err(Error::InvalidAnswer),
    }
    if !facts.attribution_reset {
        return Ok(Decision::ResetAttribution);
    }
    let Some(attribution) = &facts.attribution else {
        return Ok(Decision::ReadAttribution);
    };
    let model_attribution::Decision::Complete { groups, .. } =
        model_attribution::decide(attribution)
    else {
        return Err(Error::InvalidAnswer);
    };
    if !facts.attribution_stamped {
        return Ok(Decision::StampAttribution { groups });
    }
    if !groups.is_empty() {
        if !facts.group_budgets_read {
            return Ok(Decision::ReadGroupBudgets { groups });
        }
        for facts in &facts.group_budgets {
            let budget = common_budgets::check(facts);
            if budget.read_spend {
                return Err(Error::InvalidAnswer);
            }
            if budget.denied {
                return complete(false, true);
            }
        }
    }
    complete(true, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn facts() -> Facts {
        Facts {
            router_present: true,
            user_present: true,
            team_present: true,
            ..Facts::default()
        }
    }

    #[rstest]
    #[case::router(false, true, true, Decision::Complete { usable: false, log_failure: false })]
    #[case::user(true, false, true, Decision::Complete { usable: false, log_failure: false })]
    #[case::team(true, true, false, Decision::Complete { usable: false, log_failure: false })]
    #[case::ready(true, true, true, Decision::ReadMembership)]
    fn prerequisites(
        facts: Facts,
        #[case] router: bool,
        #[case] user: bool,
        #[case] team: bool,
        #[case] expected: Decision,
    ) {
        assert_eq!(
            decide(&Facts {
                router_present: router,
                user_present: user,
                team_present: team,
                ..facts
            })
            .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::key(0, Entity::Key)]
    #[case::user(1, Entity::User)]
    #[case::enduser(2, Entity::EndUser)]
    #[case::team(3, Entity::Team)]
    #[case::member(4, Entity::TeamMember)]
    fn budgets(facts: Facts, #[case] count: usize, #[case] expected: Entity) {
        let facts = Facts {
            membership: Membership::Loaded,
            price: Some(budget_pricing::ModelPriceFacts {
                cached: Some(false),
                lookup_failed: false,
                group_found: false,
                input_price: None,
                output_price: None,
                deployments: None,
            }),
            budget_models: Some(vec!["model".into()]),
            budget_scopes: Some(
                Entity::VARIANTS
                    .iter()
                    .map(|entity| BudgetScope {
                        entity: *entity,
                        identity_present: true,
                        configured: true,
                    })
                    .collect(),
            ),
            budgets: Entity::VARIANTS
                .iter()
                .take(count)
                .map(|entity| BudgetRead {
                    model_index: 0,
                    entity: *entity,
                    snapshot: model_budget_data::Snapshot {
                        max_budget: None,
                        local_spend: None,
                        legacy_spend: None,
                        shared_spend: None,
                        shared_started_at: None,
                        shared_window_seconds: None,
                        now: 0.0,
                    },
                })
                .collect(),
            ..facts
        };
        assert_eq!(
            decide(&facts).unwrap(),
            Decision::ReadBudget {
                model_index: 0,
                entity: expected
            }
        );
    }
}
