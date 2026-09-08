use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{Request, StatusCode};
use litellm_ai_gateway::io::realtime_pool::RealtimePool;
use litellm_ai_gateway::routes;
use litellm_ai_gateway::state::AppState;
use litellm_core::Error;
use litellm_core::router::{Deployment, LiteLLMParams, Router as ModelRouter};
use litellm_python_interop::run_async;
use pyo3::prelude::*;
use serde::Serialize;
use serde_json::Value;
use tower::ServiceExt;

use crate::errors::core_error_to_pyerr;

#[derive(Debug, Serialize)]
struct GatewayResponse {
    status: u16,
    body: Value,
}

async fn messages_request(
    model_alias: String,
    provider_model: String,
    api_base: String,
    body: Value,
) -> Result<GatewayResponse, Error> {
    let state = AppState {
        router: Arc::new(ModelRouter::new(vec![Deployment {
            model_name: model_alias,
            litellm_params: LiteLLMParams {
                model: provider_model,
                api_key: Some("trace-provider-key".to_string()),
                api_base: Some(api_base),
            },
        }])),
        master_key: Some(Arc::from("trace-master-key")),
        loggers: Arc::new(Vec::new()),
        realtime_pool: RealtimePool::disabled(),
    };
    let request = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(AUTHORIZATION, "Bearer trace-master-key")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let response = routes::app(state)
        .oneshot(request)
        .await
        .map_err(|error| match error {})?;
    let status: StatusCode = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .map_err(|error| Error::InvalidResponse(error.to_string()))?;
    let body = serde_json::from_slice(&bytes).map_err(|error| {
        Error::InvalidResponse(format!("gateway returned invalid JSON: {error}"))
    })?;
    Ok(GatewayResponse {
        status: status.as_u16(),
        body,
    })
}

#[pyfunction]
fn gateway_messages<'py>(
    py: Python<'py>,
    model_alias: String,
    provider_model: String,
    api_base: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] body: Value,
) -> PyResult<Bound<'py, PyAny>> {
    let future = messages_request(model_alias, provider_model, api_base, body);
    run_async(py, super::capture(future), core_error_to_pyerr)
}

pub(crate) fn register_gateway(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gateway_messages, module)?)
}
