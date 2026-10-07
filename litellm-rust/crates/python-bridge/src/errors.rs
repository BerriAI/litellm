//! The one exception a native call raises into Python: `RustFailure(report)`, where the report
//! says where the call failed, what went wrong and the message. Python decodes it once, in
//! `litellm.rust_bridge.failures`, and decides there whether to reroute or which public
//! exception to raise.

use std::fmt::Display;

use litellm_host::failure::{Classify, Failure, Kind, Report, Stage};
use litellm_host_python::to_py;
use pyo3::prelude::*;

pyo3::create_exception!(
    _native,
    RustFailure,
    pyo3::exceptions::PyException,
    "A native call failed. The one argument is the failure report: `stage` (where), `kind` (what) and `message`."
);

pub(crate) fn failure_to_pyerr<E: Classify + Display>(failure: Failure<E>) -> PyErr {
    report_to_pyerr(failure.report())
}

pub(crate) fn report_to_pyerr(report: Report) -> PyErr {
    Python::attach(|py| match to_py(py, &report) {
        Ok(report) => RustFailure::new_err(report),
        Err(error) => error,
    })
}

/// A capability this build does not provide, found before any work was done.
pub(crate) fn unsupported(message: impl Into<String>) -> PyErr {
    report_to_pyerr(Report {
        stage: Stage::Prepare,
        kind: Kind::Unsupported,
        message: message.into(),
    })
}

/// The public LiteLLM exception for a native failure, built once in Python from the report so
/// callbacks and the caller see the same exception. Anything that is not a `RustFailure` comes
/// back unchanged.
pub(crate) fn public_error(
    py: Python<'_>,
    native: PyErr,
    request: &Bound<'_, PyAny>,
    provider: Option<&str>,
) -> PyResult<PyErr> {
    let mapped = py
        .import("litellm.rust_bridge.failures")?
        .getattr("public_exception")?
        .call1((native.value(py), request, provider))?;
    Ok(PyErr::from_value(mapped))
}

#[cfg(test)]
mod tests {
    use litellm_host::failure::UpstreamResponse;
    use litellm_inference::RouteError;
    use pyo3::types::PyDict;

    use super::*;

    fn report<'py>(py: Python<'py>, error: &PyErr) -> Bound<'py, PyDict> {
        assert!(error.is_instance_of::<RustFailure>(py), "{error}");
        error
            .value(py)
            .getattr("args")
            .unwrap()
            .get_item(0)
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap()
    }

    #[rstest::rstest]
    fn an_upstream_failure_reports_its_stage_kind_and_provider_answer() {
        Python::initialize();
        Python::attach(|py| {
            let failure = Failure::at(
                Stage::Upstream,
                RouteError::Upstream(UpstreamResponse {
                    status: 429,
                    headers: vec![("retry-after".into(), "7".into())],
                    body: "slow down".into(),
                    url: Some("https://upstream.invalid/v1/responses".into()),
                }),
            );
            let error = failure_to_pyerr(failure);
            let report = report(py, &error);
            assert_eq!(
                report
                    .get_item("stage")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "upstream"
            );
            let kind = report.get_item("kind").unwrap().unwrap();
            assert_eq!(
                kind.get_item("kind").unwrap().extract::<String>().unwrap(),
                "upstream"
            );
            assert_eq!(
                kind.get_item("status").unwrap().extract::<u16>().unwrap(),
                429
            );
            assert_eq!(
                kind.get_item("url").unwrap().extract::<String>().unwrap(),
                "https://upstream.invalid/v1/responses"
            );
            assert_eq!(
                kind.get_item("headers")
                    .unwrap()
                    .extract::<Vec<(String, String)>>()
                    .unwrap(),
                vec![("retry-after".to_string(), "7".to_string())]
            );
            assert_eq!(
                report
                    .get_item("message")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "upstream request failed with status 429: slow down"
            );
        });
    }

    #[rstest::rstest]
    #[case::rejected_request(Failure::at(Stage::Prepare, RouteError::InvalidRequest("top_k".into())), "prepare", "request")]
    #[case::unreachable(Failure::at(Stage::Send, RouteError::Transport(litellm_http::transport::Error::Connect("refused".into()))), "send", "connection")]
    #[case::undecodable(Failure::at(Stage::Receive, RouteError::InvalidResponse("bad json".into())), "receive", "response")]
    #[case::host_fault(Failure::host(RouteError::InvalidRequest("abandoned".into())), "host", "request")]
    fn route_failures_report_stage_and_kind(
        #[case] failure: Failure<RouteError>,
        #[case] stage: &str,
        #[case] kind: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let error = failure_to_pyerr(failure);
            let report = report(py, &error);
            assert_eq!(
                report
                    .get_item("stage")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                stage
            );
            assert_eq!(
                report
                    .get_item("kind")
                    .unwrap()
                    .unwrap()
                    .get_item("kind")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                kind
            );
        });
    }

    #[test]
    fn an_unsupported_capability_is_a_prepare_failure_nothing_was_sent_for() {
        Python::initialize();
        Python::attach(|py| {
            let error = unsupported("tokenizer backend requires the tiktoken feature");
            let report = report(py, &error);
            assert_eq!(
                report
                    .get_item("stage")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "prepare"
            );
            assert_eq!(
                report
                    .get_item("kind")
                    .unwrap()
                    .unwrap()
                    .get_item("kind")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "unsupported"
            );
        });
    }
}
