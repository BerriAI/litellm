use litellm_http::transport::Error as TransportError;
use litellm_inference::RouteError;
use litellm_llms::base_llm::ocr::error::Error as OcrError;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

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

pub(crate) fn route_error_to_pyerr(error: RouteError) -> PyErr {
    match error {
        RouteError::Transport(TransportError::Http { status, body }) => {
            Python::attach(|py| upstream_error(py, status, body, Vec::new()))
                .unwrap_or_else(|error| error)
        }
        other => by_fault(other.is_request(), other.to_string()),
    }
}

pub(crate) fn upstream_error(
    py: Python<'_>,
    status: u16,
    body: String,
    headers: Vec<(String, String)>,
) -> PyResult<PyErr> {
    let error = RustUpstreamError::new_err((status, body));
    error.value(py).setattr("headers", headers)?;
    Ok(error)
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum NativeFailure {
    #[error(transparent)]
    Inference(RouteError),
    #[error(transparent)]
    Messages(RouteError),
    #[error(transparent)]
    Ocr(OcrError),
}

pub(crate) fn native_failure(
    py: Python<'_>,
    failure: NativeFailure,
    map: impl FnOnce(PyErr) -> PyErr,
) -> PyResult<PyErr> {
    let source = match &failure {
        NativeFailure::Inference(RouteError::Secret(error))
        | NativeFailure::Messages(RouteError::Secret(error)) => Some(error.source_error()),
        NativeFailure::Ocr(OcrError::Secret(error)) => Some(error.as_ref()),
        _ => None,
    };
    if let Some(original) = source.and_then(|source| crate::secrets::python_error(py, source)) {
        return Ok(original);
    }
    let native = match failure {
        NativeFailure::Inference(error) => Ok(route_error_to_pyerr(error)),
        NativeFailure::Messages(error) => messages_error(py, error),
        NativeFailure::Ocr(error) => Ok(crate::routes::ocr::errors::to_pyerr(error)),
    }?;
    Ok(map(native))
}

fn messages_error(py: Python<'_>, error: RouteError) -> PyResult<PyErr> {
    let message = match error {
        RouteError::InvalidRequest(message) => message.to_string(),
        RouteError::MissingField(field) => format!("missing required field: {field}"),
        other => return Ok(route_error_to_pyerr(other)),
    };
    let failure = PyValueError::new_err(message);
    failure.value(py).setattr("messages_request_error", true)?;
    Ok(failure)
}

/// A request the caller got wrong is a `ValueError`; anything else is a `RuntimeError`.
pub(crate) fn by_fault(is_request: bool, message: String) -> PyErr {
    if is_request {
        PyValueError::new_err(message)
    } else {
        PyRuntimeError::new_err(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest::rstest]
    #[case::unsupported(RouteError::Unsupported("test capability"), true)]
    #[case::invalid_provider(RouteError::InvalidProvider("unknown".into()), true)]
    #[case::invalid_request(RouteError::InvalidRequest("empty messages".into()), true)]
    #[case::connection(TransportError::Connect("unreachable".into()).into(), false)]
    #[case::network(TransportError::Network("timed out".into()).into(), false)]
    #[case::invalid_response(RouteError::InvalidResponse("missing usage".into()), false)]
    fn route_failures_are_terminal(#[case] error: RouteError, #[case] is_request: bool) {
        Python::initialize();
        Python::attach(|py| {
            let failure = route_error_to_pyerr(error);
            assert!(!failure.is_instance_of::<RustBridgeDeclined>(py));
            assert_eq!(failure.is_instance_of::<PyValueError>(py), is_request);
            assert_eq!(failure.is_instance_of::<PyRuntimeError>(py), !is_request);
        });
    }

    #[rstest::rstest]
    fn transport_status_survives_python_mapping() {
        Python::initialize();
        Python::attach(|py| {
            let upstream = route_error_to_pyerr(
                TransportError::Http {
                    status: 429,
                    body: "slow down".into(),
                }
                .into(),
            );
            assert!(upstream.is_instance_of::<RustUpstreamError>(py));
            assert_eq!(
                upstream
                    .value(py)
                    .getattr("args")
                    .unwrap()
                    .extract::<(u16, String)>()
                    .unwrap(),
                (429, "slow down".into())
            );
        });
    }

    #[test]
    fn missing_api_key_stays_a_runtime_error_while_other_auth_failures_are_value_errors() {
        Python::initialize();
        Python::attach(|py| {
            let missing =
                route_error_to_pyerr(RouteError::Auth(litellm_auth::Error::MissingApiKey {
                    provider: "Anthropic",
                    environment_variable: "ANTHROPIC_API_KEY",
                }));
            assert!(missing.is_instance_of::<PyRuntimeError>(py));
            let invalid =
                route_error_to_pyerr(RouteError::Auth(litellm_auth::Error::InvalidHeader));
            assert!(invalid.is_instance_of::<PyValueError>(py));
        });
    }
    #[rstest]
    #[case::rejected_request(RouteError::InvalidRequest("does not support top_k=5".into()), true)]
    #[case::missing_field(RouteError::MissingField("max_tokens"), true)]
    #[case::unresolvable_provider(RouteError::InvalidProvider("openai".into()), false)]
    #[case::upstream_failure(
        RouteError::Transport(TransportError::Http { status: 400, body: "bad".into() }),
        false,
    )]
    fn only_request_rejections_carry_the_request_error_marker(
        #[case] error: RouteError,
        #[case] marked: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let native = native_failure(py, NativeFailure::Messages(error), |error| error).unwrap();
            let marker = native
                .value(py)
                .getattr_opt("messages_request_error")
                .unwrap()
                .map(|value| value.extract::<bool>().unwrap());
            assert_eq!(marker.unwrap_or(false), marked);
        });
    }
}
