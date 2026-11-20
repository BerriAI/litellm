/// `ACTIVE_TEAMS_KEY_PREFIX` and the model: one sorted set per model.
pub(super) fn key(model: &str) -> String {
    format!("dynamic_rate_limiter:active_teams:{model}")
}
