//! A budget row as the auth cache holds it, nested in the object that names it.

use litellm_python_compat::{json, truthy::truthy};
use litellm_redis_store::auth_cache::{BudgetEntry, BudgetFullEntry, ExcludedIfNone};
use litellm_repositories::{Budget, RawJson};

use crate::{
    Error,
    column::{instant, nullable, time},
};

/// What `LiteLLM_BudgetTableFull.model_dump(mode="json")` holds for `budget`: [`budget_entry`]
/// with the reset and creation times, refused for the same columns.
pub(crate) fn budget_full_entry(budget: Budget) -> Result<BudgetFullEntry, Error> {
    let budget_reset_at = time("budget_reset_at", budget.budget_reset_at)?;
    let created_at = instant("created_at", budget.created_at)?;
    Ok(BudgetFullEntry {
        budget: budget_entry(budget)?,
        budget_reset_at,
        created_at,
    })
}

/// What `LiteLLM_BudgetTable.model_dump(mode="json")` holds for `budget`, or the column the model
/// would refuse the row for. The model refuses any `budget_limits` Python finds truthy, since only
/// customer hierarchies take concurrent budget windows.
pub(crate) fn budget_entry(budget: Budget) -> Result<BudgetEntry, Error> {
    let Budget {
        budget_id,
        name: _,
        max_budget,
        rollover_max_budget,
        soft_budget,
        max_parallel_requests,
        tpm_limit,
        rpm_limit,
        tpd_limit,
        model_max_budget,
        model_tpm_limit,
        model_rpm_limit,
        budget_duration,
        budget_limits,
        cap_policy: _,
        budget_reset_at: _,
        budget_reset_count: _,
        allowed_models,
        temp_budget_increase,
        temp_budget_expiry,
        created_at: _,
        created_by: _,
        updated_at: _,
        updated_by: _,
    } = budget;
    no_windows(budget_limits.as_ref())?;
    Ok(BudgetEntry {
        budget_id: Some(budget_id),
        soft_budget,
        max_budget,
        rollover_max_budget: ExcludedIfNone(rollover_max_budget),
        max_parallel_requests: max_parallel_requests.map(i64::from),
        tpm_limit,
        rpm_limit,
        tpd_limit,
        model_max_budget: nullable("model_max_budget", model_max_budget.as_ref())?,
        model_tpm_limit: nullable("model_tpm_limit", model_tpm_limit.as_ref())?,
        model_rpm_limit: nullable("model_rpm_limit", model_rpm_limit.as_ref())?,
        budget_duration,
        allowed_models,
        temp_budget_increase,
        temp_budget_expiry: time("temp_budget_expiry", temp_budget_expiry)?,
        unread: Vec::new(),
    })
}

fn no_windows(budget_limits: Option<&RawJson>) -> Result<(), Error> {
    match budget_limits.map(|raw| json::loads(raw.get())) {
        None => Ok(()),
        Some(Ok(windows)) if !truthy(&windows) => Ok(()),
        Some(_) => Err(Error::row("budget_limits")),
    }
}
