mod operation;

pub(super) use operation::{Failure, FailureKind, FailureStage, delete, rotate, write};

use litellm_core_utils::exception_mapping_utils::{
    TimeoutPhase, timeout_message as transport_timeout_message,
};
use litellm_error::Report;
use litellm_secrets::hashicorp::RawOperationError;
use pyo3::prelude::*;

use super::mutation::{error_value, http_message, json_value};
use super::provider::read_timeout_seconds;

const HTTP_TRANSPORT_MODULE: &str = "litellm.rust_bridge.host.http_transport";

pub(super) fn failure_value(
    py: Python<'_>,
    failure: Failure,
    context: &ErrorContext,
) -> PyResult<Py<PyAny>> {
    let message = match (&*failure.kind, failure.stage) {
        (FailureKind::Native(error), stage) if let Some(response) = error.http_failure() => {
            match stage {
                FailureStage::Current(name) if response.status == 404 => {
                    format!("Current secret {name} not found")
                }
                FailureStage::Replacement(name) if response.status == 404 => {
                    format!("Failed to verify new secret {name}")
                }
                FailureStage::Current(_) => format!(
                    "HTTP error occurred while checking current secret: {}",
                    response_text(py, &response.body)?
                ),
                FailureStage::Replacement(_) => format!(
                    "HTTP error occurred while verifying new secret: {}",
                    response_text(py, &response.body)?
                ),
                FailureStage::Mutation => {
                    http_message(py, &response.method, &response.url, response.status)?
                }
            }
        }
        (FailureKind::ValueMismatch { expected, actual }, _) => {
            let actual = json_value(py, actual)?;
            format!(
                "New secret value mismatch. Expected: {}, Got: {}",
                expected.expose(),
                actual.bind(py).str()?
            )
        }
        (kind, stage) => {
            let message = cause_message(py, kind, context)?;
            match stage {
                FailureStage::Current(_) => {
                    format!("Error checking current secret: {message}")
                }
                FailureStage::Replacement(_) => {
                    format!("Error verifying new secret: {message}")
                }
                FailureStage::Mutation => message,
            }
        }
    };
    error_value(py, message)
}

fn cause_message(py: Python<'_>, kind: &FailureKind, context: &ErrorContext) -> PyResult<String> {
    Ok(match kind {
        FailureKind::Native(error) => native_message(py, error, context)?,
        FailureKind::UnsafeName(name) => {
            format!("Invalid secret_name {}", name.into_pyobject(py)?.repr()?)
        }
        FailureKind::MissingGet(value) => {
            let value = json_value(py, value)?;
            match value.bind(py).getattr("get") {
                Err(error) => error.value(py).str()?.extract()?,
                Ok(_) => "HashiCorp Vault response payload is malformed".to_owned(),
            }
        }
        FailureKind::Json(body) => match json_value(py, body) {
            Err(error) => error.value(py).str()?.extract()?,
            Ok(_) => "HashiCorp Vault response payload is malformed".to_owned(),
        },
        FailureKind::ValueMismatch { .. } => "New secret value mismatch".to_owned(),
    })
}

fn native_message(
    py: Python<'_>,
    error: &RawOperationError,
    context: &ErrorContext,
) -> PyResult<String> {
    if let Some(error) = error.local() {
        return Ok(Report::new(error).to_string());
    }
    if let Some((method, elapsed)) = error.timeout() {
        return timeout_message(py, method, elapsed, context);
    }
    if let Some(response) = error.http_failure() {
        return http_message(py, &response.method, &response.url, response.status);
    }
    if let Some(login) = error.authentication() {
        let message = match login.source.login_status() {
            Some(status) => http_message(py, "POST", login.url, status)?,
            None => Report::new(login.source).to_string(),
        };
        let mechanism = if login.certificate {
            "TLS cert"
        } else {
            "AppRole"
        };
        return Ok(format!(
            "Could not authenticate to Vault via {mechanism}: {message}"
        ));
    }
    Ok(match error.transport() {
        Some(error) if error.status() == Some(408) => String::new(),
        Some(error) if error.is_connect() => "All connection attempts failed".to_owned(),
        Some(_) | None => "HashiCorp Vault request failed".to_owned(),
    })
}

fn timeout_message(
    py: Python<'_>,
    method: &str,
    elapsed: std::time::Duration,
    context: &ErrorContext,
) -> PyResult<String> {
    if method != "POST" {
        return Ok(if context.aiohttp {
            "Timeout on reading data from socket".to_owned()
        } else {
            String::new()
        });
    }
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item(
        "message",
        transport_timeout_message(
            TimeoutPhase::Read,
            context.timeout,
            Some(elapsed.as_secs_f64()),
        ),
    )?;
    kwargs.set_item("model", "default-model-name")?;
    kwargs.set_item("llm_provider", "litellm-httpx-handler")?;
    py.import("litellm.exceptions")?
        .getattr("Timeout")?
        .call((), Some(&kwargs))?
        .str()?
        .extract()
}

fn response_text(py: Python<'_>, body: &[u8]) -> PyResult<String> {
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("content", pyo3::types::PyBytes::new(py, body))?;
    py.import("httpx")?
        .getattr("Response")?
        .call((200,), Some(&kwargs))?
        .getattr("text")?
        .extract()
}

#[derive(Default)]
pub(super) struct ErrorContext {
    timeout: Option<f64>,
    aiohttp: bool,
}

impl ErrorContext {
    pub(super) fn capture(
        py: Python<'_>,
        system: litellm_secrets::KeyManagementSystem,
        timeout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if system != litellm_secrets::KeyManagementSystem::HashicorpVault {
            return Ok(Self::default());
        }
        Ok(Self {
            timeout: read_timeout_seconds(timeout)?,
            aiohttp: py
                .import(HTTP_TRANSPORT_MODULE)?
                .getattr("uses_aiohttp_transport")?
                .call0()?
                .extract()?,
        })
    }
}
