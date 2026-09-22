use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_cache::ExactCacheContext;
use litellm_cache_response::{CacheControls, CacheKeyInput, ResponseCacheRequest};
use litellm_host_python::from_py;
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestInput {
    key: CacheKeyInput,
    controls: Option<CacheControls>,
    ttl_seconds: Option<f64>,
    max_age_seconds: Option<f64>,
    messages: Option<Value>,
    input: Option<Value>,
    metadata: Option<Value>,
    litellm_metadata: Option<Value>,
    litellm_params: Option<Value>,
    scope: Option<String>,
}

#[derive(Clone)]
pub(super) struct NativeRequest {
    pub(super) key: CacheKeyInput,
    pub(super) controls: CacheControls,
    pub(super) ttl: Option<Duration>,
    pub(super) max_age: Option<Duration>,
    pub(super) messages: Option<Value>,
    pub(super) input: Option<Value>,
    pub(super) metadata: Option<Value>,
    pub(super) litellm_metadata: Option<Value>,
    pub(super) litellm_params: Option<Value>,
    pub(super) scope: Option<String>,
}

pub(super) fn request(value: &Bound<'_, PyAny>) -> PyResult<NativeRequest> {
    let input: RequestInput = from_py(value)?;
    request_input(input)
}

fn request_input(input: RequestInput) -> PyResult<NativeRequest> {
    let controls = input.controls.unwrap_or_else(|| {
        ResponseCacheRequest::<ExactCacheContext>::new(input.key.clone()).controls
    });
    Ok(NativeRequest {
        key: input.key,
        controls,
        ttl: input.ttl_seconds.map(duration).transpose()?,
        max_age: input.max_age_seconds.map(duration).transpose()?,
        messages: input.messages,
        input: input.input,
        metadata: input.metadata,
        litellm_metadata: input.litellm_metadata,
        litellm_params: input.litellm_params,
        scope: input.scope,
    })
}

pub(super) fn requests(value: &Bound<'_, PyAny>) -> PyResult<Vec<NativeRequest>> {
    from_py::<Vec<RequestInput>>(value)?
        .into_iter()
        .map(request_input)
        .collect()
}

pub(super) fn duration(seconds: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(seconds)
        .map_err(|_| PyValueError::new_err("cache durations must be finite and nonnegative"))
}

pub(super) fn now() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}
