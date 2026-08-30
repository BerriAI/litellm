//! Admin routes for querying models, keys, users, teams, and spend logs.

mod key_info;
mod models;
mod spend_logs;
mod team_info;
#[cfg(test)]
mod tests;
mod user_info;

use axum::Router;

use crate::state::AppState;

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new()
        .merge(models::router())
        .merge(key_info::router())
        .merge(spend_logs::router())
        .merge(user_info::router())
        .merge(team_info::router())
}
