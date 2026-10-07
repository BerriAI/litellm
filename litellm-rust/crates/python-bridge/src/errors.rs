//! The one exception a native call raises into Python: `RustFailure(report)`, where the report
//! says where the call failed, what went wrong and the message.
//!
//! A failure is settled here, before the lifecycle's terminal event: when a standby
//! implementation can serve the call and the provider never acted on it, the attempt is
//! abandoned and the bare report crosses; otherwise the public LiteLLM exception is built
//! once, in `litellm.rust_bridge.failures`, and the failure hooks see that same exception.

use std::fmt::Display;

use litellm_host::failure::{Classify, Failure, Kind, Report, Stage};
use litellm_host_python::{Settlement, to_py};
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

/// What the Python host does with a native failure. `standby` says another implementation
/// of this call is waiting; it serves the call only when the provider never acted on it.
pub(crate) fn settle<E: Classify + Display>(
    py: Python<'_>,
    failure: Failure<E>,
    standby: bool,
    request: &Bound<'_, PyAny>,
    provider: Option<&str>,
) -> PyResult<Settlement> {
    let report = failure.report();
    if standby && report.is_reroutable() {
        return Ok(Settlement::Abandon(report_to_pyerr(report)));
    }
    public_error(py, report_to_pyerr(report), request, provider).map(Settlement::Fail)
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
    use litellm_host::failure::{Exchange, UpstreamResponse};
    use litellm_http::transport::Error as TransportError;
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

    fn field(report: &Bound<'_, PyDict>, name: &str) -> String {
        report
            .get_item(name)
            .unwrap()
            .unwrap()
            .extract::<String>()
            .unwrap()
    }

    fn kind(report: &Bound<'_, PyDict>) -> String {
        report
            .get_item("kind")
            .unwrap()
            .unwrap()
            .get_item("kind")
            .unwrap()
            .extract::<String>()
            .unwrap()
    }

    fn upstream(status: u16) -> UpstreamResponse {
        UpstreamResponse {
            status,
            headers: vec![("retry-after".into(), "7".into())],
            body: "slow down".into(),
            url: Some("https://upstream.invalid/v1/responses".into()),
        }
    }

    fn request(py: Python<'_>) -> Bound<'_, PyDict> {
        let request = PyDict::new(py);
        request.set_item("model", "openai/gpt-5").unwrap();
        request
    }

    #[rstest::rstest]
    fn an_upstream_failure_reports_its_stage_kind_and_provider_answer() {
        Python::initialize();
        Python::attach(|py| {
            let error = failure_to_pyerr(Failure::<RouteError>::from(Exchange::Rejected(
                upstream(429),
            )));
            let report = report(py, &error);
            assert_eq!(field(&report, "stage"), "upstream");
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
                field(&report, "message"),
                "upstream request failed with status 429: slow down"
            );
        });
    }

    #[rstest::rstest]
    #[case::rejected_request(Failure::prepare(RouteError::InvalidRequest("top_k".into())), "prepare", "request")]
    #[case::unreachable(Failure::from(Exchange::Unreached(RouteError::Transport(TransportError::Connect("refused".into())))), "send", "connection")]
    #[case::undecodable(Failure::receive(RouteError::InvalidResponse("bad json".into())), "receive", "response")]
    #[case::machine_fault(Failure::receive(RouteError::Machine(litellm_host::machine::MachineFault::Abandoned)), "receive", "internal")]
    fn route_failures_report_stage_and_kind(
        #[case] failure: Failure<RouteError>,
        #[case] stage: &str,
        #[case] expected_kind: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let error = failure_to_pyerr(failure);
            let report = report(py, &error);
            assert_eq!(field(&report, "stage"), stage);
            assert_eq!(kind(&report), expected_kind);
        });
    }

    #[test]
    fn an_unsupported_capability_is_a_prepare_failure_nothing_was_sent_for() {
        Python::initialize();
        Python::attach(|py| {
            let error = unsupported("tokenizer backend requires the tiktoken feature");
            let report = report(py, &error);
            assert_eq!(field(&report, "stage"), "prepare");
            assert_eq!(kind(&report), "unsupported");
        });
    }

    #[rstest::rstest]
    #[case::rejected_request(Failure::prepare(RouteError::InvalidRequest("top_k".into())))]
    #[case::unreachable(Failure::from(Exchange::Unreached(RouteError::Transport(TransportError::Connect("refused".into())))))]
    #[case::provider_400(Failure::from(Exchange::Rejected(upstream(400))))]
    fn with_a_standby_a_failure_the_provider_never_acted_on_abandons_the_attempt_bare(
        #[case] failure: Failure<RouteError>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let Settlement::Abandon(error) = settle(py, failure, true, &request(py), None).unwrap()
            else {
                panic!("expected the attempt to be abandoned");
            };
            assert!(error.is_instance_of::<RustFailure>(py), "{error}");
        });
    }

    #[rstest::rstest]
    #[case::provider_429(Failure::from(Exchange::Rejected(upstream(429))), true, "RateLimitError")]
    #[case::provider_500(Failure::from(Exchange::Rejected(upstream(500))), true, "InternalServerError")]
    #[case::undecodable(Failure::receive(RouteError::InvalidResponse("bad json".into())), true, "APIError")]
    #[case::no_standby_rejected_request(Failure::prepare(RouteError::InvalidRequest("top_k".into())), false, "BadRequestError")]
    #[case::no_standby_provider_400(Failure::from(Exchange::Rejected(upstream(400))), false, "BadRequestError")]
    fn otherwise_the_failure_is_the_public_exception_with_the_report_as_its_cause(
        #[case] failure: Failure<RouteError>,
        #[case] standby: bool,
        #[case] public: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let Settlement::Fail(error) = settle(py, failure, standby, &request(py), None).unwrap()
            else {
                panic!("expected the failure to reach the caller");
            };
            let litellm = py.import("litellm").unwrap();
            assert!(
                error.is_instance(py, &litellm.getattr(public).unwrap()),
                "{error}"
            );
            assert!(
                error
                    .value(py)
                    .getattr("__cause__")
                    .unwrap()
                    .is_instance_of::<RustFailure>()
            );
        });
    }
}
