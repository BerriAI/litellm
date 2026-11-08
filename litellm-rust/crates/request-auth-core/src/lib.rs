#![forbid(unsafe_code)]

pub mod credential;
pub mod budget_pricing;
pub mod common_budgets;
mod error;
pub mod fallback_access;
pub mod fallback_candidate;
mod key;
pub mod key_model_access;
pub mod model_access;
pub mod model_attribution;
pub mod model_budget_data;
mod model_budgets;
pub mod model_groups;
pub mod org_routes;
pub mod patterns;
pub mod personal_budget;
pub mod personal_models;
pub mod team_override;
pub mod model_helpers;
pub mod personal_owner;
pub mod personal_model_request;
pub mod member_models;
pub mod resolved_models;
pub mod owner_teams;
pub mod protocol;
pub mod route_access;
pub mod unpriced_models;

pub use error::Error;
pub use key::{start, start_resolved_key};
pub use model_budgets::{
    check_model_budgets, check_team_member_model_budgets, check_user_model_budgets,
};
