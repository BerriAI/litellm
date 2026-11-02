//! The stream-recovery entry points of `RouterCoreHandle`: what may still take over a stream that
//! failed before content, the same-group stream retry, and the fallback walk re-entered from a
//! stream. The host's stream wrappers call them; the calls are driven like `start`.

use litellm_router_core::{
    FailureKind, Op, Router,
    call::{StreamOrigin, StreamPolicy},
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use super::{RouterCall, attempt_outcome, config, config_error_to_pyerr, document, host::Finish};

/// A stream recovery a host wrapper asks for.
#[derive(Clone, Copy)]
pub(super) enum Recovery {
    /// `_astream_retry_same_group`.
    Retry,
    /// The fallback walk re-entered from the stream wrapper.
    Fallback,
}

fn origin(value: &Bound<'_, PyAny>) -> PyResult<StreamOrigin> {
    serde_json::from_value(document(value)?)
        .map_err(|error| PyValueError::new_err(format!("a stream origin: {error}")))
}

fn kind(kind: Option<&str>) -> PyResult<Option<FailureKind>> {
    kind.map(|kind| {
        kind.parse()
            .map_err(|_| PyValueError::new_err(format!("`{kind}` is not a router failure kind")))
    })
    .transpose()
}

/// `{"can_retry", "can_fall_back", "refusal_fallback", "retry_budget": int | None,
/// "policy_applies": bool}`.
pub(super) fn policy<'py>(
    py: Python<'py>,
    router: &Router,
    request: &Bound<'py, PyAny>,
    stream_origin: &Bound<'py, PyAny>,
    failure_kind: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let request = config::request(document(request)?, None).map_err(config_error_to_pyerr)?;
    let StreamPolicy {
        can_retry,
        can_fall_back,
        refusal_fallback,
        retry_budget,
    } = router.stream_policy(&request, &origin(stream_origin)?, kind(failure_kind)?);
    let dict = PyDict::new(py);
    dict.set_item("can_retry", can_retry)?;
    dict.set_item("can_fall_back", can_fall_back)?;
    dict.set_item("refusal_fallback", refusal_fallback)?;
    dict.set_item("retry_budget", retry_budget.map(|(budget, _)| budget))?;
    dict.set_item(
        "policy_applies",
        retry_budget.is_some_and(|(_, applies)| applies),
    )?;
    Ok(dict)
}

/// A recovery call for a stream from `stream_origin` that failed with `failure`.
pub(super) fn recover(
    router: &Router,
    recovery: Recovery,
    op: &str,
    request: &Bound<'_, PyAny>,
    stream_origin: &Bound<'_, PyAny>,
    failure: &Bound<'_, PyDict>,
) -> PyResult<RouterCall> {
    let op: Op = op.parse().unwrap_or_else(|_| Op::Other(op.to_owned()));
    let request = config::request(document(request)?, None).map_err(config_error_to_pyerr)?;
    let stream_origin = origin(stream_origin)?;
    let Err(failure) = attempt_outcome(failure)? else {
        return Err(PyValueError::new_err("a stream failure needs `ok: False`"));
    };
    let router = router.clone();
    Ok(RouterCall::new(async move {
        let response = match recovery {
            Recovery::Retry => {
                router
                    .retry_stream(op, request, stream_origin, failure)
                    .await
            }
            Recovery::Fallback => {
                router
                    .fallback_stream(op, request, stream_origin, failure)
                    .await
            }
        };
        response.map(Finish::Call)
    }))
}
