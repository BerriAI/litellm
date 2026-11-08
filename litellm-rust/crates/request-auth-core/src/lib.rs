#![forbid(unsafe_code)]

pub mod credential;
mod error;
pub mod fallback_access;
mod key;
pub mod key_model_access;
pub mod model_access;
mod model_budgets;
pub mod model_groups;
pub mod org_routes;
pub mod patterns;
pub mod protocol;
pub mod route_access;
pub mod unpriced_models;

pub use error::Error;
pub use key::{start, start_resolved_key};
pub use model_budgets::{
    check_model_budgets, check_team_member_model_budgets, check_user_model_budgets,
};
