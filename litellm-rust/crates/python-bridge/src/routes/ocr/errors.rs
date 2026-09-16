use litellm_core::ocr::Error;
use litellm_core::transport::Error as TransportError;
use pyo3::exceptions::{PyBaseException, PyException};
use pyo3::prelude::*;
use pyo3::types::PyDict;

enum Public {
    BadRequest,
    Authentication,
    NotFound,
    Timeout,
    RateLimit,
    InternalServer,
    BadGateway,
    ServiceUnavailable,
    Api(u16),
    ApiConnection,
}

impl Public {
    fn class_name(&self) -> &'static str {
        match self {
            Self::BadRequest => "BadRequestError",
            Self::Authentication => "AuthenticationError",
            Self::NotFound => "NotFoundError",
            Self::Timeout => "Timeout",
            Self::RateLimit => "RateLimitError",
            Self::InternalServer => "InternalServerError",
            Self::BadGateway => "BadGatewayError",
            Self::ServiceUnavailable => "ServiceUnavailableError",
            Self::Api(_) => "APIError",
            Self::ApiConnection => "APIConnectionError",
        }
    }

    fn from_status(status: u16) -> Self {
        match status {
            400 | 422 => Self::BadRequest,
            401 => Self::Authentication,
            404 => Self::NotFound,
            408 | 504 => Self::Timeout,
            429 => Self::RateLimit,
            500 => Self::InternalServer,
            502 => Self::BadGateway,
            503 => Self::ServiceUnavailable,
            other => Self::Api(other),
        }
    }
}

struct Failure {
    public: Public,
    detail: String,
    headers: Vec<(String, String)>,
}

fn classify(error: Error) -> Failure {
    match error {
        Error::Provider {
            status,
            body,
            headers,
        } => Failure {
            public: Public::from_status(status),
            detail: body,
            headers,
        },
        Error::Transport(TransportError::Http { status, body }) => Failure {
            public: Public::from_status(status),
            detail: body,
            headers: Vec::new(),
        },
        error if error.is_request() => Failure {
            public: Public::BadRequest,
            detail: error.to_string(),
            headers: Vec::new(),
        },
        error => Failure {
            public: Public::ApiConnection,
            detail: error.to_string(),
            headers: Vec::new(),
        },
    }
}

fn exception_provider(provider: &str) -> String {
    let mut chars = provider.chars();
    match chars.next() {
        Some(first) => format!("{}{}Exception", first.to_uppercase(), chars.as_str()),
        None => "Exception".into(),
    }
}

pub(super) fn public_exception(
    py: Python<'_>,
    error: Error,
    model: &str,
    provider: &str,
) -> PyResult<PyErr> {
    if let Error::FileRead { path, source } = error {
        let errno = source.raw_os_error().unwrap_or(match source.kind() {
            std::io::ErrorKind::NotFound => 2,
            std::io::ErrorKind::PermissionDenied => 13,
            _ => 5,
        });
        return Ok(PyErr::from_value(
            py.import("builtins")?.getattr("OSError")?.call1((
                errno,
                source.to_string(),
                path.into_pyobject(py)?.call_method0("__fspath__")?,
            ))?,
        ));
    }
    raise_public(py, classify(error), model, provider, None)
}

pub(super) fn public_host_exception(
    py: Python<'_>,
    error: &Py<PyBaseException>,
    model: &str,
    provider: &str,
) -> PyResult<PyErr> {
    let value = error.bind(py);
    if !value.is_instance_of::<PyException>() || is_public(value)? {
        return Ok(PyErr::from_value(value.clone().into_any()));
    }
    let failure = Failure {
        public: Public::ApiConnection,
        detail: value.str()?.to_string(),
        headers: Vec::new(),
    };
    raise_public(py, failure, model, provider, Some(value))
}

fn is_public(value: &Bound<'_, PyBaseException>) -> PyResult<bool> {
    let types = value
        .py()
        .import("litellm")?
        .getattr("LITELLM_EXCEPTION_TYPES")?;
    for public in types.try_iter()? {
        if value.is_instance(&public?)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn raise_public(
    py: Python<'_>,
    failure: Failure,
    model: &str,
    provider: &str,
    context: Option<&Bound<'_, PyBaseException>>,
) -> PyResult<PyErr> {
    let class = failure.public.class_name();
    let message = format!(
        "{}{} - {}",
        match failure.public {
            Public::RateLimit | Public::Api(_) | Public::ApiConnection => format!("{class}: "),
            _ => String::new(),
        },
        exception_provider(provider),
        failure.detail
    );
    let kwargs = PyDict::new(py);
    kwargs.set_item("message", message)?;
    kwargs.set_item("llm_provider", provider)?;
    kwargs.set_item("model", model)?;
    if let Public::Api(status) = failure.public {
        kwargs.set_item("status_code", status)?;
    }
    let value = py
        .import("litellm")?
        .getattr(class)?
        .call((), Some(&kwargs))?;
    if !failure.headers.is_empty() {
        let headers = PyDict::new(py);
        for (name, header) in &failure.headers {
            headers.set_item(name, header)?;
        }
        value.setattr("litellm_response_headers", headers)?;
    }
    if let Some(context) = context {
        value.setattr("__context__", context)?;
    }
    Ok(PyErr::from_value(value))
}

pub(super) fn to_pyerr(error: Error) -> PyErr {
    Python::attach(|py| public_exception(py, error, "", "").unwrap_or_else(|error| error))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classified(error: Error) -> (&'static str, String, Vec<(String, String)>) {
        let failure = classify(error);
        (failure.public.class_name(), failure.detail, failure.headers)
    }

    #[test]
    fn provider_status_selects_public_class_and_keeps_details() {
        let (class, detail, headers) = classified(Error::Provider {
            status: 429,
            body: "rate limited".into(),
            headers: vec![("retry-after".into(), "17".into())],
        });
        assert_eq!(class, "RateLimitError");
        assert_eq!(detail, "rate limited");
        assert_eq!(headers, [("retry-after".to_string(), "17".to_string())]);
    }

    #[test]
    fn status_table_matches_legacy_openai_mapping() {
        for (status, class) in [
            (400, "BadRequestError"),
            (401, "AuthenticationError"),
            (404, "NotFoundError"),
            (408, "Timeout"),
            (422, "BadRequestError"),
            (429, "RateLimitError"),
            (500, "InternalServerError"),
            (502, "BadGatewayError"),
            (503, "ServiceUnavailableError"),
            (504, "Timeout"),
            (418, "APIError"),
        ] {
            let (mapped, _, _) = classified(Error::Transport(TransportError::Http {
                status,
                body: "x".into(),
            }));
            assert_eq!(mapped, class, "{status}");
        }
    }

    #[test]
    fn request_shape_and_statusless_failures_use_their_public_families() {
        assert_eq!(classified(Error::RequestFormat).0, "BadRequestError");
        let (class, detail, _) = classified(Error::MissingAzureAiCredentials);
        assert_eq!(class, "APIConnectionError");
        assert!(detail.contains("Missing Azure AI credentials"));
        let (class, detail, _) = classified(Error::Transport(TransportError::Network(
            "connection reset".into(),
        )));
        assert_eq!(class, "APIConnectionError");
        assert!(detail.contains("connection reset"));
    }

    #[test]
    fn exception_provider_matches_legacy_capitalisation() {
        assert_eq!(exception_provider("mistral"), "MistralException");
        assert_eq!(exception_provider("azure_ai"), "Azure_aiException");
        assert_eq!(exception_provider(""), "Exception");
    }
}
