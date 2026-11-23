//! The stream-recovery entry points of `RouterCoreHandle`: what may still take over a stream that
//! failed before content, the same-group stream retry, and the fallback walk re-entered from a
//! stream. The host's stream wrappers call them; the calls are driven like `start`.

use litellm_router_core::{FailureKind, Op, Router, call, call::StreamPolicy};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use super::{
    RouterCall, attempt_outcome, config, config_error_to_pyerr, document,
    host::{Finish, PythonRouterHost},
};

/// A stream recovery a host wrapper asks for.
#[derive(Clone, Copy)]
pub(super) enum Recovery {
    /// `_astream_retry_same_group`.
    Retry,
    /// The fallback walk re-entered from the stream wrapper.
    Fallback,
}

/// Where a streamed response came from, as router-core built it for the attempt that returned
/// it. Python cannot take it apart: the stream's wrapper keeps it and hands it back to the
/// recovery entry points, so a take-over starts from everything the attempt's level carried.
#[pyclass(frozen, name = "StreamOrigin")]
pub(crate) struct StreamOrigin(call::StreamOrigin);

impl From<call::StreamOrigin> for StreamOrigin {
    fn from(origin: call::StreamOrigin) -> Self {
        Self(origin)
    }
}

#[pymethods]
impl StreamOrigin {
    /// The origin of a stream of `group` that no attempt of a call returned.
    #[new]
    fn outside_a_call(group: String) -> Self {
        Self(call::StreamOrigin::outside_a_call(group))
    }

    /// This origin after the stream's group retried it.
    fn retried(&self, attempted_retries: u32, committed_budget: u32) -> Self {
        Self(self.0.clone().retried(attempted_retries, committed_budget))
    }

    #[getter]
    fn group(&self) -> &str {
        self.0.group()
    }

    #[getter]
    fn depth(&self) -> u32 {
        self.0.depth()
    }

    #[getter]
    fn level(&self) -> u32 {
        self.0.level()
    }

    #[getter]
    fn deployment_id(&self) -> Option<&str> {
        self.0.deployment_id()
    }

    #[getter]
    fn attempted_retries(&self) -> u32 {
        self.0.attempted_retries()
    }
}

fn kind(kind: Option<&str>) -> PyResult<Option<FailureKind>> {
    kind.map(|kind| {
        kind.parse()
            .map_err(|_| PyValueError::new_err(format!("`{kind}` is not a router failure kind")))
    })
    .transpose()
}

/// `{"can_retry", "can_fall_back", "refusal_fallback", "overrides_params", "retry_budget": int | None,
/// "policy_applies": bool}`.
pub(super) fn policy<'py>(
    py: Python<'py>,
    router: &Router,
    request: &Bound<'py, PyAny>,
    stream_origin: &StreamOrigin,
    failure_kind: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let request = config::parse_request(document(request)?, None).map_err(config_error_to_pyerr)?;
    let StreamPolicy {
        can_retry,
        can_fall_back,
        refusal_fallback,
        retry_budget,
        overrides_params,
    } = router.stream_policy(&request, &stream_origin.0, kind(failure_kind)?);
    let dict = PyDict::new(py);
    dict.set_item("can_retry", can_retry)?;
    dict.set_item("can_fall_back", can_fall_back)?;
    dict.set_item("refusal_fallback", refusal_fallback)?;
    dict.set_item("overrides_params", overrides_params)?;
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
    stream_origin: &StreamOrigin,
    failure: &Bound<'_, PyDict>,
) -> PyResult<RouterCall> {
    let op = Op::new(op);
    let request = config::parse_request(document(request)?, None).map_err(config_error_to_pyerr)?;
    let stream_origin = stream_origin.0.clone();
    let Err(failure) = attempt_outcome(failure)? else {
        return Err(PyValueError::new_err("a stream failure needs `ok: False`"));
    };
    let router = router.clone();
    Ok(RouterCall::new(async move {
        let response = match recovery {
            Recovery::Retry => {
                router
                    .retry_stream(&PythonRouterHost, op, request, stream_origin, failure)
                    .await
            }
            Recovery::Fallback => {
                router
                    .fallback_stream(&PythonRouterHost, op, request, stream_origin, failure)
                    .await
            }
        };
        response.map(Finish::Call).map_err(|ended| ended.error)
    }))
}
