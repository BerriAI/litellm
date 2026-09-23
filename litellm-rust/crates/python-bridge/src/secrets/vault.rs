use litellm_secrets::hashicorp::{Error, PythonFailure, PythonFailureKind, PythonFailureStage};
use pyo3::prelude::*;

use super::mutation::{error_value, http_message, json_value};

pub(super) fn failure_value(
    py: Python<'_>,
    failure: PythonFailure,
    context: &ErrorContext,
) -> PyResult<Py<PyAny>> {
    let message = match *failure.kind {
        PythonFailureKind::Http {
            method,
            url,
            status,
            body,
        } => match failure.stage {
            PythonFailureStage::Current(name) if status == 404 => {
                format!("Current secret {name} not found")
            }
            PythonFailureStage::Replacement(name) if status == 404 => {
                format!("Failed to verify new secret {name}")
            }
            PythonFailureStage::Current(_) => format!(
                "HTTP error occurred while checking current secret: {}",
                response_text(py, &body)?
            ),
            PythonFailureStage::Replacement(_) => format!(
                "HTTP error occurred while verifying new secret: {}",
                response_text(py, &body)?
            ),
            PythonFailureStage::Mutation => http_message(py, &method, &url, status)?,
        },
        PythonFailureKind::ValueMismatch { expected, actual } => {
            let actual = json_value(py, &actual)?;
            format!(
                "New secret value mismatch. Expected: {}, Got: {}",
                expected.expose(),
                actual.bind(py).str()?
            )
        }
        kind => {
            let message = cause_message(py, kind, context)?;
            match failure.stage {
                PythonFailureStage::Current(_) => {
                    format!("Error checking current secret: {message}")
                }
                PythonFailureStage::Replacement(_) => {
                    format!("Error verifying new secret: {message}")
                }
                PythonFailureStage::Mutation => message,
            }
        }
    };
    error_value(py, message)
}

fn cause_message(
    py: Python<'_>,
    kind: PythonFailureKind,
    context: &ErrorContext,
) -> PyResult<String> {
    Ok(match kind {
        PythonFailureKind::Local(error) => error.to_string(),
        PythonFailureKind::UnsafeName(name) => {
            format!("Invalid secret_name {}", name.into_pyobject(py)?.repr()?)
        }
        PythonFailureKind::Timeout { method, elapsed } => {
            if method == "POST" {
                let elapsed = py
                    .import("builtins")?
                    .call_method1("round", (elapsed.as_secs_f64(), 3))?;
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item(
                    "message",
                    format!(
                        "Connection timed out. Timeout passed={}, time taken={} seconds",
                        context.timeout.as_deref().unwrap_or("None"),
                        elapsed.str()?
                    ),
                )?;
                kwargs.set_item("model", "default-model-name")?;
                kwargs.set_item("llm_provider", "litellm-httpx-handler")?;
                kwargs.set_item("headers", pyo3::types::PyDict::new(py))?;
                py.import("litellm")?
                    .getattr("Timeout")?
                    .call((), Some(&kwargs))?
                    .str()?
                    .extract()?
            } else if context.aiohttp {
                "Timeout on reading data from socket".to_owned()
            } else {
                String::new()
            }
        }
        PythonFailureKind::Transport(source) => {
            if let Some(error) = request_error(&source) {
                if error.is_timeout() {
                    String::new()
                } else if error.is_connect() {
                    "All connection attempts failed".to_owned()
                } else {
                    "HashiCorp Vault request failed".to_owned()
                }
            } else {
                "HashiCorp Vault request failed".to_owned()
            }
        }
        PythonFailureKind::MissingGet(value) => {
            let value = json_value(py, &value)?;
            match value.bind(py).getattr("get") {
                Err(error) => error.value(py).str()?.extract()?,
                Ok(_) => "HashiCorp Vault response payload is malformed".to_owned(),
            }
        }
        PythonFailureKind::Json(body) => match json_value(py, &body) {
            Err(error) => error.value(py).str()?.extract()?,
            Ok(_) => "HashiCorp Vault response payload is malformed".to_owned(),
        },
        PythonFailureKind::Authentication {
            source,
            url,
            certificate,
        } => {
            let message = match source {
                Error::LoginStatus { status } => http_message(py, "POST", &url, status)?,
                error => error.to_string(),
            };
            let mechanism = if certificate { "TLS cert" } else { "AppRole" };
            format!("Could not authenticate to Vault via {mechanism}: {message}")
        }
        PythonFailureKind::Http {
            method,
            url,
            status,
            ..
        } => http_message(py, &method, &url, status)?,
        PythonFailureKind::ValueMismatch { .. } => "New secret value mismatch".to_owned(),
    })
}

fn request_error<'a>(error: &'a (dyn std::error::Error + 'static)) -> Option<&'a reqwest::Error> {
    error
        .downcast_ref::<reqwest::Error>()
        .or_else(|| error.source().and_then(request_error))
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
    timeout: Option<String>,
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
            timeout: timeout.map(|value| value.str()?.extract()).transpose()?,
            aiohttp: py
                .import("litellm.llms.custom_httpx.http_handler")?
                .getattr("AsyncHTTPHandler")?
                .call_method0("_should_use_aiohttp_transport")?
                .extract()?,
        })
    }
}
