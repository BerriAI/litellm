use std::{collections::BTreeMap, time::SystemTime};

use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentedKey {
    Admin { expires: Option<SystemTime> },
    NonAdmin,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TeamMember {
    pub user_id: String,
    pub team_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KeyFacts {
    pub has_token: bool,
    pub blocked: bool,
    pub expires: Option<SystemTime>,
    pub spend: f64,
    pub soft_budget: Option<f64>,
    pub max_budget: Option<f64>,
    pub user_id: Option<String>,
    pub team_member: Option<TeamMember>,
    pub budget_throttle_pct: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnerFacts {
    pub deactivated: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TeamMemberBudget {
    pub spend: f64,
    pub max_budget: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyModelBudgets {
    pub fallbacks: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelBudgetScope {
    pub models: Vec<String>,
    pub rewritable_model: Option<String>,
    pub key: Option<KeyModelBudgets>,
    pub user_has_model_budgets: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TeamMemberModelBudgetScope {
    pub models: Vec<String>,
    pub rewritable_model: Option<String>,
    pub has_model_budgets: bool,
    pub can_fall_back: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FallbackTarget {
    pub requested: String,
    pub model: String,
    pub served_model: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SigningKey {
    pub key_id: String,
    pub algorithm: String,
    pub public_key_pem: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Introspection {
    pub active: bool,
    pub user_id: Value,
    pub role: Value,
    pub team_id: Value,
}
