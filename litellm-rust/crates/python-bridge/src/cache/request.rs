use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_cache::{ExactCacheContext, SemanticCacheContext};
use litellm_cache_response::{CacheControls, CacheKeyInput, ResponseCacheRequest};
use litellm_host_python::from_py;
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestInput {
    key: CacheKeyInput,
    controls: Option<CacheControls>,
    ttl_seconds: Option<f64>,
    max_age_seconds: Option<f64>,
    input: Option<Value>,
    messages: Option<Vec<Value>>,
    metadata: Option<Map<String, Value>>,
    scope: Option<String>,
}

pub(super) struct CacheRequest {
    key: CacheKeyInput,
    controls: CacheControls,
    ttl: Option<Duration>,
    max_age: Option<Duration>,
    input: Option<Value>,
    messages: Vec<Value>,
    metadata: Map<String, Value>,
    scope: Option<String>,
}

impl CacheRequest {
    pub(super) fn exact(&self) -> ResponseCacheRequest<ExactCacheContext> {
        let mut request = ResponseCacheRequest::new(self.key.clone());
        request.controls = self.controls;
        request.context.ttl = self.ttl;
        request.max_age = self.max_age;
        request
    }

    pub(super) fn semantic(&self) -> ResponseCacheRequest<SemanticCacheContext> {
        ResponseCacheRequest {
            key: self.key.clone(),
            controls: self.controls,
            context: SemanticCacheContext {
                input: self.input.clone(),
                messages: self.messages.clone(),
                metadata: self.metadata.clone(),
                scope: self.scope.clone(),
                ttl: self.ttl,
            },
            max_age: self.max_age,
        }
    }
}

pub(super) fn request(value: &Bound<'_, PyAny>) -> PyResult<CacheRequest> {
    let input: RequestInput = from_py(value)?;
    request_input(input)
}

fn request_input(input: RequestInput) -> PyResult<CacheRequest> {
    let defaults = ResponseCacheRequest::<ExactCacheContext>::new(input.key.clone());
    Ok(CacheRequest {
        key: input.key,
        controls: input.controls.unwrap_or(defaults.controls),
        ttl: input.ttl_seconds.map(duration).transpose()?,
        max_age: input.max_age_seconds.map(duration).transpose()?,
        input: input.input,
        messages: input.messages.unwrap_or_default(),
        metadata: input.metadata.unwrap_or_default(),
        scope: input.scope,
    })
}

pub(super) fn requests(value: &Bound<'_, PyAny>) -> PyResult<Vec<CacheRequest>> {
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
