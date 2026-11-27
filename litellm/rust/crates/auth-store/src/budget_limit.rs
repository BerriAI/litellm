//! A `budget_limits` window as the auth cache holds it, nested in a team or a user.

use litellm_redis_store::auth_cache::BudgetLimitEntry;

/// The windows a `budget_limits` column holds, as `BudgetLimit` reads them: without the keys the
/// model does not name, which it drops.
pub(crate) fn windows(windows: Option<Vec<BudgetLimitEntry>>) -> Option<Vec<BudgetLimitEntry>> {
    windows.map(|windows| {
        windows
            .into_iter()
            .map(|window| BudgetLimitEntry {
                unread: Vec::new(),
                ..window
            })
            .collect()
    })
}
