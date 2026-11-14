use std::sync::Arc;

use litellm_redis_client::CallTag;
use litellm_redis_store::{RequestRedis, Send};
use pyo3::prelude::*;

use super::{call, duration};
use crate::{execution::run_async_value, redis::events::RedisCall};

/// The tag of a deadline flush, asked on the thread whose post-call declaration arms the deadline.
fn deadline_flush_tag() -> Option<CallTag> {
    Python::attach(RedisCall::deadline_flush)
        .inspect_err(|error| {
            tracing::warn!(%error, "a post-call deadline flush will be reported without its request");
        })
        .ok()
        .map(RedisCall::tag)
}

/// One request's native Redis batches. Python builds one per request scope and passes it to each
/// request-aware store call.
#[pyclass(frozen, name = "NativeRequestRedis")]
pub(crate) struct NativeRequestRedis(pub(crate) RequestRedis);

#[pymethods]
impl NativeRequestRedis {
    #[new]
    fn new(py: Python<'_>, post_call_deadline_seconds: f64) -> PyResult<Self> {
        let runtime = litellm_host_python::runtime_handle()?;
        let deadline = duration(post_call_deadline_seconds)?;
        Ok(Self(crate::logger::capture(py).scope(|| {
            RequestRedis::with_deadline_tag(runtime, deadline, Arc::new(deadline_flush_tag))
        })))
    }

    fn flush_pre<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let request = self.0.clone();
        let scope = call(py, "request_redis_batch")?;
        run_async_value(py, async move {
            scope.scope(request.flush_pre()).await;
            Ok(())
        })
    }

    fn flush_post<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let request = self.0.clone();
        let scope = call(py, "post_call_redis_batch")?;
        run_async_value(py, async move {
            scope.scope(request.flush_post()).await;
            Ok(())
        })
    }
}

/// The Python face of `Send`: which batch a store op rides. Each variant carries exactly what it
/// needs, so there is no phase string and no phase without a request.
#[pyclass(frozen, name = "NativeSend")]
pub(crate) enum NativeSend {
    Now(),
    PreCall(Py<NativeRequestRedis>),
    PostCall(Py<NativeRequestRedis>),
}

impl NativeSend {
    pub(crate) fn send(&self) -> Send<'_> {
        match self {
            Self::Now() => Send::Now,
            Self::PreCall(request) => Send::PreCall(&request.get().0),
            Self::PostCall(request) => Send::PostCall(&request.get().0),
        }
    }
}
