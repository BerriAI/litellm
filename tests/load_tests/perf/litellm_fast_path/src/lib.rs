// PyO3 extension: Rust fast-path for LiteLLM's /v1/chat/completions.
//
// When a request's model is configured for a mock response, build the
// OpenAI-compatible JSON reply entirely in Rust — skipping FastAPI route
// setup, Pydantic validation, the litellm router chain, the executor hop,
// and callback dispatch. Returns response body bytes for the caller to
// hand to the ASGI send() channel.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Deserialize)]
struct ChatRequest<'a> {
    #[serde(borrow)]
    model: &'a str,
}

/// Try to build a mock response for the given request body.
///
/// Args:
///   body: Raw request body bytes.
///   mock_map: dict of {model_name: mock_response_content}.
///
/// Returns:
///   bytes: the OpenAI-compatible response JSON, OR None if no fast path applies
///          (invalid JSON, missing model, or model not in mock_map).
#[pyfunction]
fn try_build_mock_response<'py>(
    py: Python<'py>,
    body: &[u8],
    mock_map: &Bound<'py, PyDict>,
) -> PyResult<Option<Py<PyAny>>> {
    // Parse just enough to get the model name — serde with a borrowed &str
    // avoids allocating a String here.
    let req: ChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };

    // Look up the configured mock content for this model.
    let mock_content: String = match mock_map.get_item(req.model)? {
        Some(v) => v.extract()?,
        None => return Ok(None),
    };

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let resp: Value = json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": mock_content,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    });

    let bytes = serde_json::to_vec(&resp)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Return a Python `bytes` object.
    let py_bytes = pyo3::types::PyBytes::new_bound(py, &bytes);
    Ok(Some(py_bytes.into_any().unbind()))
}

#[pymodule]
fn litellm_fast_path(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(try_build_mock_response, m)?)?;
    Ok(())
}
