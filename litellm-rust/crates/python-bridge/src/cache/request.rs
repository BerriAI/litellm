use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_cache::{ExactCacheContext, SemanticCacheContext, SemanticCacheScope};
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
    messages: Option<Vec<Value>>,
    input: Option<String>,
    metadata: Option<Map<String, Value>>,
    scope: Option<SemanticCacheScope>,
}

pub(super) fn request(
    value: &Bound<'_, PyAny>,
) -> PyResult<ResponseCacheRequest<SemanticCacheContext>> {
    let input: RequestInput = from_py(value)?;
    request_input(input)
}

fn request_input(input: RequestInput) -> PyResult<ResponseCacheRequest<SemanticCacheContext>> {
    let mut request: ResponseCacheRequest<SemanticCacheContext> =
        ResponseCacheRequest::new(input.key);
    if let Some(controls) = input.controls {
        request.controls = controls;
    }
    request.context.ttl = input.ttl_seconds.map(duration).transpose()?;
    request.context.messages = input.messages.unwrap_or_default();
    request.context.input = input.input;
    request.context.metadata = input.metadata.unwrap_or_default();
    request.context.scope = input.scope.unwrap_or_default();
    request.max_age = input.max_age_seconds.map(duration).transpose()?;
    Ok(request)
}

pub(super) fn requests(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<ResponseCacheRequest<SemanticCacheContext>>> {
    from_py::<Vec<RequestInput>>(value)?
        .into_iter()
        .map(request_input)
        .collect()
}

pub(super) fn exact(
    request: &ResponseCacheRequest<SemanticCacheContext>,
) -> ResponseCacheRequest<ExactCacheContext> {
    request.clone().with_context(ExactCacheContext {
        ttl: request.context.ttl,
    })
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
