use litellm_core::error::Error as CoreError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use strum::{AsRefStr, Display};
use thiserror::Error as ThisError;

#[derive(Clone, Copy, Debug, Display, PartialEq, Eq)]
pub(crate) enum Route {
    #[strum(serialize = "chat completions")]
    ChatCompletions,
    #[strum(serialize = "messages")]
    Messages,
    #[strum(serialize = "OCR")]
    Ocr,
}

#[derive(Clone, Copy, Debug, Display, PartialEq, Eq)]
pub(crate) enum StateSlot {
    #[strum(serialize = "logging state")]
    Logging,
    #[strum(serialize = "pre-call state")]
    PreCall,
    #[strum(serialize = "roots")]
    Roots,
}

#[derive(Debug, ThisError)]
pub(crate) enum Error {
    #[error("{0} lifecycle is complete")]
    LifecycleComplete(Route),
    #[error("{0} request was already sent or cleared")]
    RequestConsumed(Route),
    #[error("{0} {1} was cleared")]
    StateCleared(Route, StateSlot),
    #[error("{0} terminal record is unavailable")]
    TerminalUnavailable(Route),
    #[error("{0}")]
    Declined(String),
    #[error("{message}")]
    Upstream { status: u16, message: String },
    #[error("core bridge error")]
    Core(CoreError),
    #[error("chat completions bridge error")]
    ChatCompletions(CoreError),
    #[error("messages provider bridge error")]
    MessagesProvider(CoreError),
    #[error("OCR provider bridge error")]
    Ocr {
        error: CoreError,
        model: String,
        provider: String,
    },
    #[error(transparent)]
    Python(#[from] PyErr),
}

impl Error {
    pub(crate) fn declined(message: impl Into<String>) -> Self {
        Self::Declined(message.into())
    }

    pub(crate) fn upstream(status: u16, message: impl Into<String>) -> Self {
        Self::Upstream {
            status,
            message: message.into(),
        }
    }

    fn into_pyerr(self, py: Python<'_>) -> PyErr {
        match self {
            error @ (Self::LifecycleComplete(_)
            | Self::RequestConsumed(_)
            | Self::StateCleared(_, _)
            | Self::TerminalUnavailable(_)) => PyRuntimeError::new_err(error.to_string()),
            Self::Declined(message) => RustBridgeDeclined::new_err(message),
            Self::Upstream { status, message } => RustUpstreamError::new_err((status, message)),
            Self::Core(error) => generic_core_error_to_pyerr(error),
            Self::ChatCompletions(error) => chat_completions_core_error_to_pyerr(py, error),
            Self::MessagesProvider(error) => messages_provider_core_error_to_pyerr(py, error),
            Self::Ocr {
                error,
                model,
                provider,
            } => ocr_core_error_to_pyerr(py, error, &model, &provider),
            Self::Python(error) => error,
        }
    }
}

pyo3::create_exception!(
    _native,
    RustBridgeDeclined,
    pyo3::exceptions::PyException,
    "The route declined before calling the provider, so the host may retry on its own path."
);

pyo3::create_exception!(
    _native,
    RustUpstreamError,
    pyo3::exceptions::PyException,
    "The provider call was already issued and failed. Args are (status, message); status is 0 when there was no HTTP response."
);

pyo3::create_exception!(
    _native,
    RustBridgeDriverError,
    pyo3::exceptions::PyRuntimeError,
    "The Rust bridge could not initialize a route driver."
);

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        Python::attach(|py| error.into_pyerr(py))
    }
}

fn generic_core_error_to_pyerr(error: CoreError) -> PyErr {
    match error {
        CoreError::InvalidProvider(_) => PyValueError::new_err("Invalid provider configuration"),
        CoreError::InvalidRequest(_) => PyValueError::new_err("Invalid provider request"),
        CoreError::InvalidType { .. } | CoreError::MissingField(_) => {
            PyValueError::new_err(error.to_string())
        }
        CoreError::Auth(_) => PyValueError::new_err("Provider authentication failed"),
        CoreError::Http { status, .. } => {
            PyRuntimeError::new_err(format!("Provider request failed (HTTP {status})"))
        }
        CoreError::Network(_) | CoreError::Connect(_) => {
            PyRuntimeError::new_err("Provider transport failed")
        }
        CoreError::InvalidResponse(_) => PyRuntimeError::new_err("Invalid provider response"),
        CoreError::Routing(_) => PyRuntimeError::new_err("Provider routing failed"),
        CoreError::Unsupported(_) => PyRuntimeError::new_err("Operation is not supported"),
    }
}

fn chat_completions_core_error_to_pyerr(py: Python<'_>, error: CoreError) -> PyErr {
    match error {
        CoreError::Unsupported(_) => Error::declined("Operation is not supported").into_pyerr(py),
        CoreError::Auth(_) => Error::declined("Provider authentication failed").into_pyerr(py),
        CoreError::InvalidProvider(_) => {
            Error::declined("Invalid provider configuration").into_pyerr(py)
        }
        CoreError::InvalidRequest(_) => Error::declined("Invalid provider request").into_pyerr(py),
        CoreError::InvalidType { .. } | CoreError::MissingField(_) => {
            Error::declined(error.to_string()).into_pyerr(py)
        }
        CoreError::Routing(_) => Error::declined("Provider routing failed").into_pyerr(py),
        CoreError::Connect(_) => Error::declined("Could not reach provider").into_pyerr(py),
        CoreError::Http { status, .. } => {
            Error::upstream(status, format!("Provider request failed (HTTP {status})"))
                .into_pyerr(py)
        }
        CoreError::Network(_) => Error::upstream(0, "Provider transport failed").into_pyerr(py),
        CoreError::InvalidResponse(_) => {
            Error::upstream(0, "Invalid provider response").into_pyerr(py)
        }
    }
}

fn messages_provider_core_error_to_pyerr(py: Python<'_>, error: CoreError) -> PyErr {
    match error {
        CoreError::Http { status, .. } => {
            Error::upstream(status, format!("Provider request failed (HTTP {status})"))
                .into_pyerr(py)
        }
        CoreError::Network(_) | CoreError::Connect(_) => {
            Error::upstream(0, "Provider transport failed").into_pyerr(py)
        }
        CoreError::InvalidResponse(_) => {
            Error::upstream(0, "Invalid provider response").into_pyerr(py)
        }
        error => generic_core_error_to_pyerr(error),
    }
}

#[derive(AsRefStr, Clone, Copy, Debug, PartialEq, Eq)]
enum OcrExceptionClass {
    #[strum(serialize = "BadRequestError")]
    BadRequest,
    #[strum(serialize = "AuthenticationError")]
    Authentication,
    #[strum(serialize = "PermissionDeniedError")]
    PermissionDenied,
    #[strum(serialize = "NotFoundError")]
    NotFound,
    #[strum(serialize = "UnprocessableEntityError")]
    UnprocessableEntity,
    #[strum(serialize = "RateLimitError")]
    RateLimit,
    #[strum(serialize = "InternalServerError")]
    InternalServer,
    #[strum(serialize = "BadGatewayError")]
    BadGateway,
    #[strum(serialize = "ServiceUnavailableError")]
    ServiceUnavailable,
    #[strum(serialize = "APIError")]
    Api,
}

impl From<u16> for OcrExceptionClass {
    fn from(status: u16) -> Self {
        match status {
            400 => Self::BadRequest,
            401 => Self::Authentication,
            403 => Self::PermissionDenied,
            404 => Self::NotFound,
            422 => Self::UnprocessableEntity,
            429 => Self::RateLimit,
            500 => Self::InternalServer,
            502 => Self::BadGateway,
            503 => Self::ServiceUnavailable,
            _ => Self::Api,
        }
    }
}

fn ocr_sdk_error(py: Python<'_>, status: u16, model: &str, provider: &str) -> Result<PyErr, Error> {
    let class = OcrExceptionClass::from(status);
    let kwargs = PyDict::new(py);
    kwargs.set_item(
        pyo3::intern!(py, "message"),
        format!("OCR provider request failed (HTTP {status})"),
    )?;
    kwargs.set_item(pyo3::intern!(py, "model"), model)?;
    kwargs.set_item(pyo3::intern!(py, "llm_provider"), provider)?;
    if class == OcrExceptionClass::Api {
        kwargs.set_item(pyo3::intern!(py, "status_code"), status)?;
    } else {
        let httpx = py.import("httpx")?;
        let request = httpx
            .getattr(pyo3::intern!(py, "Request"))?
            .call1(("POST", "https://litellm.ai"))?;
        let response_kwargs = PyDict::new(py);
        response_kwargs.set_item(pyo3::intern!(py, "request"), request)?;
        let response = httpx
            .getattr(pyo3::intern!(py, "Response"))?
            .call((status,), Some(&response_kwargs))?;
        kwargs.set_item(pyo3::intern!(py, "response"), response)?;
    }
    let instance = py
        .import("litellm.exceptions")?
        .getattr(class.as_ref())?
        .call((), Some(&kwargs))?;
    Ok(PyErr::from_value(instance))
}

fn ocr_core_error_to_pyerr(py: Python<'_>, error: CoreError, model: &str, provider: &str) -> PyErr {
    let status = match error {
        CoreError::Unsupported(message) => return PyRuntimeError::new_err(message),
        CoreError::Auth(_) => 401,
        CoreError::Http { status, .. } => status,
        CoreError::Network(_) | CoreError::Connect(_) => {
            return PyRuntimeError::new_err("OCR transport failed");
        }
        CoreError::InvalidResponse(_) => {
            return PyRuntimeError::new_err("Invalid OCR provider response");
        }
        error => return generic_core_error_to_pyerr(error),
    };
    ocr_sdk_error(py, status, model, provider).unwrap_or_else(Into::into)
}

pub(crate) fn ocr_python_error_to_pyerr(
    py: Python<'_>,
    error: PyErr,
    model: &str,
    provider: &str,
    arguments: &Bound<'_, PyDict>,
) -> PyErr {
    if !error.is_instance_of::<pyo3::exceptions::PyException>(py) {
        return error;
    }
    let mapped = (|| -> PyResult<PyErr> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("model", model)?;
        kwargs.set_item("custom_llm_provider", provider)?;
        kwargs.set_item("original_exception", error.value(py))?;
        kwargs.set_item("completion_kwargs", arguments)?;
        kwargs.set_item("extra_kwargs", arguments)?;
        py.import("litellm")?
            .getattr("exception_type")?
            .call((), Some(&kwargs))
            .map(PyErr::from_value)
    })()
    .unwrap_or_else(|error| error);
    mapped.set_context(py, Some(error));
    mapped
}

pub(crate) fn core_error_to_pyerr(error: CoreError) -> PyErr {
    Error::Core(error).into()
}

pub(crate) fn chat_completions_error_to_pyerr(error: CoreError) -> PyErr {
    Error::ChatCompletions(error).into()
}

pub(crate) fn messages_provider_error_to_pyerr(error: CoreError) -> PyErr {
    Error::MessagesProvider(error).into()
}

pub(crate) fn ocr_error_to_pyerr(
    py: Python<'_>,
    error: CoreError,
    model: &str,
    provider: &str,
) -> PyErr {
    Error::Ocr {
        error,
        model: model.to_owned(),
        provider: provider.to_owned(),
    }
    .into_pyerr(py)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("RustBridgeDeclined", py.get_type::<RustBridgeDeclined>())?;
    module.add("RustUpstreamError", py.get_type::<RustUpstreamError>())?;
    module.add(
        "RustBridgeDriverError",
        py.get_type::<RustBridgeDriverError>(),
    )
}

#[cfg(test)]
#[path = "../tests/unit/errors.rs"]
mod tests;

pub(crate) fn ocr_preparation_error_to_pyerr(
    py: Python<'_>,
    error: CoreError,
    auth_error: Option<PyErr>,
    model: &str,
    custom_llm_provider: Option<&str>,
    arguments: &Bound<'_, PyDict>,
) -> PyErr {
    let resolved =
        litellm_core::routing_utils::provider::get_custom_llm_provider(model, custom_llm_provider);
    let model = resolved.as_ref().map_or(model, |value| value.model);
    let provider = resolved
        .as_ref()
        .map_or("", |value| value.custom_llm_provider);
    if let Some(error) = auth_error {
        return ocr_python_error_to_pyerr(py, error, model, provider, arguments);
    }
    if provider == "azure_ai" {
        if let CoreError::InvalidRequest(message) = &error {
            return ocr_python_error_to_pyerr(
                py,
                PyValueError::new_err(message.clone()),
                model,
                provider,
                arguments,
            );
        }
    }
    ocr_error_to_pyerr(py, error, model, provider)
}
