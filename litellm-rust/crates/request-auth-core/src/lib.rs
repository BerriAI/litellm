#![forbid(unsafe_code)]

pub mod access;
pub mod admission;
pub mod budget_pricing;
pub mod common_budgets;
pub mod credential;
mod error;
mod key;
pub mod managed;
pub mod model_budget_data;
mod model_budgets;
pub mod patterns;
pub mod route_access;
pub mod org_routes;
pub mod model_access;
pub mod model_groups;
pub mod fallback_access;
pub mod key_model_access;
pub mod protocol;
pub mod unpriced_models;

pub use error::Error;
pub use key::{start, start_resolved_key};
pub use model_budgets::{
    check_model_budgets, check_team_member_model_budgets, check_user_model_budgets,
};
