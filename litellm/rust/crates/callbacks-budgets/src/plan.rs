use litellm_callbacks::{BudgetPlan, BudgetPlanCounter};
use litellm_redis_store::spend_counters::{NotASpendCounter, SpendCounter};

#[derive(Clone, Debug, PartialEq)]
pub struct PlannedCounter {
    pub counter_key: String,
    pub max_budget: f64,
    pub fallback_spend: f64,
    pub entity_type: String,
    pub entity_id: String,
    pub spend_log_entity_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub counters: Vec<PlannedCounter>,
    pub reservation_cost: f64,
    pub input_cost: f64,
    pub input_tokens: Option<i64>,
    pub throttle_counter: Option<String>,
    pub fail_closed: bool,
    pub status_code: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingMember(pub &'static str);

impl std::fmt::Display for MissingMember {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the budget plan has no {}", self.0)
    }
}

fn required<T>(value: Option<T>, member: &'static str) -> Result<T, MissingMember> {
    value.ok_or(MissingMember(member))
}

impl TryFrom<BudgetPlanCounter> for PlannedCounter {
    type Error = MissingMember;

    fn try_from(counter: BudgetPlanCounter) -> Result<Self, Self::Error> {
        Ok(Self {
            counter_key: required(counter.counter_key, "counters[].counter_key")?,
            max_budget: required(counter.max_budget, "counters[].max_budget")?,
            fallback_spend: required(counter.fallback_spend, "counters[].fallback_spend")?,
            entity_type: required(counter.entity_type, "counters[].entity_type")?,
            entity_id: required(counter.entity_id, "counters[].entity_id")?,
            spend_log_entity_id: counter.spend_log_entity_id,
        })
    }
}

impl TryFrom<BudgetPlan> for Plan {
    type Error = MissingMember;

    fn try_from(plan: BudgetPlan) -> Result<Self, Self::Error> {
        Ok(Self {
            counters: required(plan.counters, "counters")?
                .into_iter()
                .map(PlannedCounter::try_from)
                .collect::<Result<_, _>>()?,
            reservation_cost: required(plan.reservation_cost, "reservation_cost")?,
            input_cost: required(plan.input_cost, "input_cost")?,
            input_tokens: plan.input_tokens,
            throttle_counter: plan.throttle_counter,
            fail_closed: plan.fail_closed.unwrap_or_default(),
            status_code: required(plan.status_code, "status_code")?,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Counter {
    pub planned: PlannedCounter,
    pub counter: SpendCounter,
}

impl Counter {
    pub fn key(&self) -> &str {
        &self.planned.counter_key
    }
}

impl Plan {
    pub(crate) fn counters(&self) -> Result<Vec<Counter>, NotASpendCounter> {
        self.counters
            .iter()
            .map(|planned| {
                Ok(Counter {
                    counter: SpendCounter::parse(&planned.counter_key)?,
                    planned: planned.clone(),
                })
            })
            .collect()
    }
}

pub(crate) fn entity_type_value(entity_type: &str) -> Option<&'static str> {
    match entity_type {
        "Key" => Some("key"),
        "Team" => Some("team"),
        "TeamMember" => Some("team_member"),
        "User" => Some("user"),
        "EndUser" => Some("end_user"),
        "Tag" => Some("tag"),
        "Model access group" => Some("model_access_group"),
        "Organization" => Some("organization"),
        "Project" => Some("project"),
        "customer_org" => Some("customer_org"),
        "customer_team" => Some("customer_team"),
        "customer_member" => Some("customer_member"),
        _ => None,
    }
}
