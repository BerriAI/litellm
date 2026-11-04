pub(crate) mod request;

use litellm_host_python::PythonContext;
use pyo3::prelude::*;

use crate::redis::events::RedisCall;

/// The labels `RedisCache` reports for this operation, in the caller's context so its
/// `service_target` reaches the service event.
fn call(py: Python<'_>, call_type: &'static str) -> PyResult<RedisCall> {
    Ok(RedisCall {
        context: PythonContext::capture(py)?,
        call_type,
        caller: None,
    })
}
