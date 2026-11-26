use litellm_python_compat::datetime::IsoDateTime;

use super::record::{ExcludedIfNone, record, required};

record! {
    /// One window of a team's `budget_limits`, as `BudgetLimit` dumps it, nested in
    /// [`TeamEntry`](super::TeamEntry).
    pub struct BudgetLimitEntry {
        budget_duration: String = required!("budget_duration"),
        max_budget: f64 = required!("max_budget"),
        reset_at: Option<IsoDateTime>,
        window_start: ExcludedIfNone<IsoDateTime>,
    }
}
