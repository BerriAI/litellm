//! The retry decisions `RouterCoreHandle` exposes as members: `get_num_retries_from_retry_policy`
//! and `should_retry_this_error`, decided by router-core for a failure Python describes the way it
//! describes a failed attempt.

use litellm_router_core::{
    AttemptError, Router,
    retry::{RetryContext, resolve_retry_policy, should_retry as core_should_retry},
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

fn failure(outcome: &Bound<'_, PyDict>) -> PyResult<AttemptError> {
    match super::attempt_outcome(outcome)? {
        Err(failure) => Ok(failure),
        Ok(_) => Err(PyValueError::new_err("a failure needs `ok: False`")),
    }
}

pub(crate) fn policy_retries(
    router: &Router,
    outcome: &Bound<'_, PyDict>,
    model_group: Option<&str>,
) -> PyResult<Option<u32>> {
    let failure = failure(outcome)?;
    let settings = router.settings();
    let policy = match model_group {
        Some(group) => resolve_retry_policy(
            settings.retry_policy.as_ref(),
            group,
            Some(&settings.model_group_retry_policy),
        ),
        None => settings.retry_policy.as_ref(),
    };
    Ok(policy.and_then(|policy| policy.retries_for(&failure)))
}

pub(crate) fn allowed_fails(router: &Router, outcome: &Bound<'_, PyDict>) -> PyResult<Option<u32>> {
    Ok(router.settings().allowed_fails_for(&failure(outcome)?))
}

pub(crate) fn should_retry(
    outcome: &Bound<'_, PyDict>,
    (healthy, all): (usize, usize),
    (context_window, content_policy, regular): (bool, bool, bool),
) -> PyResult<bool> {
    Ok(core_should_retry(
        &failure(outcome)?,
        RetryContext {
            healthy,
            all,
            has_context_window_fallbacks: context_window,
            has_content_policy_fallbacks: content_policy,
            has_regular_fallbacks: regular,
        },
    ))
}
