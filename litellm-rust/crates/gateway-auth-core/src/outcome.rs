use crate::model_budgets::WithinModelBudgets;

#[derive(Clone, Debug, PartialEq)]
pub struct ModelBudgetExceeded {
    pub message: String,
    pub current_cost: f64,
    pub max_budget: f64,
    pub entity_type: Option<String>,
    pub entity_id: Option<String>,
}

pub type ModelBudgetResult = Result<WithinModelBudgets, ModelBudgetExceeded>;

#[derive(Clone, Debug, PartialEq)]
pub enum Denial {
    UserAuthDisabled,
    MasterKeyOnlyRoute,
    TeamMemberBudgetExceeded {
        user_id: String,
        team_id: String,
        spend: f64,
        max_budget: f64,
    },
    KeyBudgetExceeded {
        spend: f64,
        max_budget: f64,
    },
    ModelBudgetExceeded(ModelBudgetExceeded),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Allow { budget_throttle_pct: Option<f64> },
    Deny(Denial),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::VariantArray)]
pub enum ExemptionReason {
    PublicRoute,
    PassThroughAuthOff,
    NoAuthDevMode,
    CustomAuth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestCheck {
    Exempt(ExemptionReason),
    Checked,
}
