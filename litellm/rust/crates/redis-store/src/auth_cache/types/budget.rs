use litellm_python_compat::datetime::IsoDateTime;

use super::{
    bag::{Bag, Ints},
    record::{ExcludedIfNone, record},
};

record! {
    /// `LiteLLM_BudgetTable.model_dump(mode="json")`, nested under `litellm_budget_table` in
    /// [`ProjectEntry`](super::ProjectEntry).
    pub struct BudgetEntry {
        budget_id: Option<String>,
        soft_budget: Option<f64>,
        max_budget: Option<f64>,
        rollover_max_budget: ExcludedIfNone<f64>,
        max_parallel_requests: Option<i64>,
        tpm_limit: Option<i64>,
        rpm_limit: Option<i64>,
        tpd_limit: Option<i64>,
        model_max_budget: Option<Bag>,
        model_tpm_limit: Option<Ints>,
        model_rpm_limit: Option<Ints>,
        budget_duration: Option<String>,
        allowed_models: Option<Vec<String>>,
        temp_budget_increase: Option<f64>,
        temp_budget_expiry: Option<IsoDateTime>,
    }
}
