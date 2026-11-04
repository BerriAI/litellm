use std::time::Duration;

use litellm_redis_store::RequestRedis;
use pyo3::prelude::*;

use super::call;
use crate::execution::run_async_value;

/// One request's native Redis batches. Python builds one per request scope and passes it to each
/// request-aware store call.
#[pyclass(frozen, name = "NativeRequestRedis")]
pub(crate) struct NativeRequestRedis(pub(crate) RequestRedis);

#[pymethods]
impl NativeRequestRedis {
    #[new]
    fn new(post_call_deadline_seconds: f64) -> PyResult<Self> {
        Ok(Self(RequestRedis::new(
            litellm_host_python::runtime_handle()?,
            Duration::from_secs_f64(post_call_deadline_seconds.max(0.0)),
        )))
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
