use std::collections::BTreeMap;

use litellm_core::batches::{
    Connection, CreateBatchRequest, Error, RetrieveBatchRequest, create_batch as run_create_batch,
    http_client, retrieve_batch as run_retrieve_batch,
};
use litellm_http::transport::Error as TransportError;
use litellm_llms::anthropic::batches::transformation::LiteLlmMessageBatch;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{
    errors::{RustBridgeDeclined, RustUpstreamError},
    logger::{run_async, run_sync},
    marshal::optional_timeout,
};

struct OwnedConnection {
    api_key: Option<String>,
    api_base: Option<String>,
    extra_headers: Option<BTreeMap<String, String>>,
    timeout_seconds: Option<f64>,
}

impl OwnedConnection {
    fn borrow(&self) -> Connection<'_> {
        Connection {
            api_key: self.api_key.as_deref(),
            api_base: self.api_base.as_deref(),
            extra_headers: self.extra_headers.clone().unwrap_or_default().into_iter().collect(),
            timeout: optional_timeout(self.timeout_seconds),
        }
    }
}

fn env_lookup(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

async fn retrieve(batch_id: String, connection: OwnedConnection) -> Result<LiteLlmMessageBatch, Error> {
    run_retrieve_batch(
        http_client(),
        RetrieveBatchRequest {
            batch_id: &batch_id,
            connection: connection.borrow(),
        },
        &env_lookup,
    )
    .await
}

async fn create(
    input_jsonl: String,
    model: Option<String>,
    connection: OwnedConnection,
) -> Result<LiteLlmMessageBatch, Error> {
    run_create_batch(
        http_client(),
        CreateBatchRequest {
            model: model.as_deref(),
            input_jsonl: &input_jsonl,
            connection: connection.borrow(),
        },
        &env_lookup,
    )
    .await
}

fn upstream_error(error: TransportError) -> PyErr {
    match error {
        TransportError::Http { status, body } => RustUpstreamError::new_err((status, body)),
        TransportError::Network(message) | TransportError::Connect(message) => {
            RustUpstreamError::new_err((0u16, message))
        }
    }
}

/// Python keeps a retrieve implementation, so anything that fails before the
/// request goes out declines and Python raises its own error for it.
fn retrieve_error_to_pyerr(error: Error) -> PyErr {
    match error {
        Error::Request(_) | Error::Transport(TransportError::Connect(_)) => {
            RustBridgeDeclined::new_err(error.to_string())
        }
        Error::Transport(error) => upstream_error(error),
        Error::InvalidResponse(_) => RustUpstreamError::new_err((0u16, error.to_string())),
    }
}

/// Python has no Anthropic create to fall back to, so a rejected request is
/// the caller's error.
fn create_error_to_pyerr(error: Error) -> PyErr {
    match error {
        Error::Request(_) => PyValueError::new_err(error.to_string()),
        Error::Transport(error) => upstream_error(error),
        Error::InvalidResponse(_) => RustUpstreamError::new_err((0u16, error.to_string())),
    }
}

#[pyfunction]
#[pyo3(signature = (batch_id, api_key=None, api_base=None, extra_headers=None, timeout_seconds=None))]
pub(crate) fn retrieve_batch(
    py: Python<'_>,
    batch_id: String,
    api_key: Option<String>,
    api_base: Option<String>,
    extra_headers: Option<BTreeMap<String, String>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let connection = OwnedConnection {
        api_key,
        api_base,
        extra_headers,
        timeout_seconds,
    };
    run_sync(py, retrieve(batch_id, connection), retrieve_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (batch_id, api_key=None, api_base=None, extra_headers=None, timeout_seconds=None))]
pub(crate) fn aretrieve_batch(
    py: Python<'_>,
    batch_id: String,
    api_key: Option<String>,
    api_base: Option<String>,
    extra_headers: Option<BTreeMap<String, String>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'_, PyAny>> {
    let connection = OwnedConnection {
        api_key,
        api_base,
        extra_headers,
        timeout_seconds,
    };
    run_async(py, retrieve(batch_id, connection), retrieve_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (input_jsonl, model=None, api_key=None, api_base=None, extra_headers=None, timeout_seconds=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per Python keyword"
)]
pub(crate) fn create_batch(
    py: Python<'_>,
    input_jsonl: String,
    model: Option<String>,
    api_key: Option<String>,
    api_base: Option<String>,
    extra_headers: Option<BTreeMap<String, String>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let connection = OwnedConnection {
        api_key,
        api_base,
        extra_headers,
        timeout_seconds,
    };
    run_sync(py, create(input_jsonl, model, connection), create_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (input_jsonl, model=None, api_key=None, api_base=None, extra_headers=None, timeout_seconds=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per Python keyword"
)]
pub(crate) fn acreate_batch(
    py: Python<'_>,
    input_jsonl: String,
    model: Option<String>,
    api_key: Option<String>,
    api_base: Option<String>,
    extra_headers: Option<BTreeMap<String, String>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'_, PyAny>> {
    let connection = OwnedConnection {
        api_key,
        api_base,
        extra_headers,
        timeout_seconds,
    };
    run_async(py, create(input_jsonl, model, connection), create_error_to_pyerr)
}

#[cfg(test)]
mod tests {
    use litellm_llms::base_llm::chat::transformation::Error as LlmError;
    use rstest::rstest;

    use super::*;

    #[derive(Debug, PartialEq)]
    enum Raised {
        Declined(String),
        Upstream(u16, String),
        Value(String),
    }

    fn raised(error: PyErr) -> Raised {
        Python::initialize();
        Python::attach(|py| {
            if error.is_instance_of::<RustBridgeDeclined>(py) {
                return Raised::Declined(error.value(py).to_string());
            }
            if error.is_instance_of::<PyValueError>(py) {
                return Raised::Value(error.value(py).to_string());
            }
            assert!(error.is_instance_of::<RustUpstreamError>(py), "{error}");
            let (status, message): (u16, String) = error.value(py).getattr("args").unwrap().extract().unwrap();
            Raised::Upstream(status, message)
        })
    }

    fn request_error() -> Error {
        Error::Request(LlmError::InvalidRequest("bad".into()))
    }

    fn http_error() -> Error {
        Error::Transport(TransportError::Http {
            status: 404,
            body: "missing".into(),
        })
    }

    fn connect_error() -> Error {
        Error::Transport(TransportError::Connect("refused".into()))
    }

    fn invalid_response() -> Error {
        Error::InvalidResponse("[]".into())
    }

    #[rstest]
    #[case::request(request_error(), Raised::Declined("invalid request: bad".into()))]
    #[case::connect(connect_error(), Raised::Declined("could not reach the provider: refused".into()))]
    #[case::http(http_error(), Raised::Upstream(404, "missing".into()))]
    #[case::network(Error::Transport(TransportError::Network("reset".into())), Raised::Upstream(0, "reset".into()))]
    #[case::invalid_response(invalid_response(), Raised::Upstream(0, "invalid Anthropic batch response: []".into()))]
    fn retrieve_declines_only_before_the_request_is_sent(#[case] error: Error, #[case] expected: Raised) {
        assert_eq!(raised(retrieve_error_to_pyerr(error)), expected);
    }

    #[rstest]
    #[case::request(request_error(), Raised::Value("invalid request: bad".into()))]
    #[case::connect(connect_error(), Raised::Upstream(0, "refused".into()))]
    #[case::http(http_error(), Raised::Upstream(404, "missing".into()))]
    #[case::invalid_response(invalid_response(), Raised::Upstream(0, "invalid Anthropic batch response: []".into()))]
    fn create_never_declines(#[case] error: Error, #[case] expected: Raised) {
        assert_eq!(raised(create_error_to_pyerr(error)), expected);
    }
}
