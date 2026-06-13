use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use guardrails::{
    GuardrailInput, GuardrailOutcome, GuardrailStatus, InputType, ProviderConfig, RequestContext,
    Verdict,
};
use guardrails::HttpClient;

static HTTP_CLIENT: OnceLock<Arc<HttpClient>> = OnceLock::new();

fn http_client() -> &'static Arc<HttpClient> {
    HTTP_CLIENT.get_or_init(|| Arc::new(HttpClient::new(Duration::from_secs(10))))
}

#[derive(Serialize)]
struct ApplyResponse {
    verdict: Verdict,
    provider_response: serde_json::Value,
    guardrail_status: GuardrailStatus,
    duration_ms: u64,
}

fn build_response(result: Result<GuardrailOutcome, String>, duration_ms: u64) -> ApplyResponse {
    match result {
        Ok(outcome) => ApplyResponse {
            guardrail_status: outcome.status(),
            verdict: outcome.verdict,
            provider_response: outcome.provider_response,
            duration_ms: outcome.duration_ms,
        },
        Err(message) => ApplyResponse {
            guardrail_status: GuardrailStatus::GuardrailFailedToRespond,
            verdict: Verdict::Block {
                violation_message: message,
                detections: vec![],
            },
            provider_response: serde_json::Value::Null,
            duration_ms,
        },
    }
}

#[derive(Deserialize)]
struct ApplyRequest {
    config: ProviderConfig,
    input: GuardrailInput,
    input_type: InputType,
    #[serde(default)]
    context: RequestContext,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[pyfunction]
fn apply_guardrail<'py>(py: Python<'py>, request_json: String) -> PyResult<Bound<'py, PyAny>> {
    let http = http_client().clone();

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let req: ApplyRequest = serde_json::from_str(&request_json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid request JSON: {e}"))
        })?;

        let provider = guardrails::build(req.config, &http).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid config: {e}"))
        })?;

        let timeout = Duration::from_millis(req.timeout_ms.unwrap_or(5000));
        let start = Instant::now();

        let result = tokio::time::timeout(
            timeout,
            provider.apply(&req.input, req.input_type, &req.context),
        )
        .await;

        let duration_ms = start.elapsed().as_millis() as u64;

        let response = match result {
            Ok(Ok(outcome)) => build_response(Ok(outcome), duration_ms),
            Ok(Err(provider_err)) => build_response(
                Err(format!("Guardrail unavailable: {provider_err}")),
                duration_ms,
            ),
            Err(_elapsed) => build_response(Err("Guardrail timed out".to_owned()), duration_ms),
        };

        serde_json::to_string(&response).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to serialize response: {e}"
            ))
        })
    })
}

#[pymodule]
fn litellm_guardrails_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_guardrail, m)?)?;
    Ok(())
}
