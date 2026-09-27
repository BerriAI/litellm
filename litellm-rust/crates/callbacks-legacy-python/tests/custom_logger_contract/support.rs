use std::{
    convert::Infallible,
    ffi::CString,
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex, MutexGuard, OnceLock},
    time::Duration,
};

use bytes::Bytes;
use litellm_callbacks_legacy_python::{
    LegacySurface, PassThroughStream, PublicCall, run_legacy_call,
};
use litellm_core::{
    messages::{
        Error, MessagesCall, MessagesShaping,
        route::{BODY_FIELDS, Messages, MessagesOutput, MessagesStreamHead, messages_machine},
    },
    resources::CoreResources,
};
use litellm_host_python::{InvokeError, ProtocolHost, json_fields, to_py};
use litellm_http::{HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver};
use litellm_secrets::{SecretValue, source::SecretSource};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyBytes, PyDict, PyTuple},
};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

pub const MODEL: &str = "callback-contract-model";

pub fn response() -> Value {
    json!({
        "id": "msg_contract", "type": "message", "role": "assistant", "model": MODEL,
        "content": [{"type": "text", "text": "original"}], "stop_reason": "end_turn",
        "stop_sequence": null, "usage": {"input_tokens": 7, "output_tokens": 3}
    })
}

pub fn stream_events() -> String {
    [
        json!({"type": "message_start", "message": {"id": "msg_contract", "type": "message", "role": "assistant", "model": MODEL, "content": [], "stop_reason": null, "usage": {"input_tokens": 7, "output_tokens": 0}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "original"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
        json!({"type": "message_stop"}),
    ].iter().map(|event| format!("event: {}\ndata: {event}\n\n", event["type"].as_str().unwrap())).collect()
}

fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| tokio::runtime::Runtime::new().unwrap())
}

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Sync,
    Async,
}
impl Mode {
    pub fn asynchronous(self) -> bool {
        matches!(self, Self::Async)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Upstream {
    Success,
    Failure,
    Stream,
}

struct NoSecrets;
impl SecretSource for NoSecrets {
    fn get_secret_str<'a>(
        &'a self,
        _: &'a str,
    ) -> Pin<
        Box<dyn Future<Output = Result<Option<SecretValue>, litellm_secrets::Error>> + Send + 'a>,
    > {
        Box::pin(async { Ok(None) })
    }
}

struct MessagesHost {
    error: Py<PyAny>,
}
impl ProtocolHost for MessagesHost {
    type Protocol = Messages;
    type Failure = PyErr;

    fn project(
        &mut self,
        _py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<MessagesCall, InvokeError<Error>> {
        let body = litellm_core::messages::messages_body(json_fields(
            ["model"].into_iter().chain(BODY_FIELDS),
            |name| arguments.get_item(name),
        )?)
        .map_err(InvokeError::Native)?;
        Ok(MessagesCall {
            body,
            api_key: Some("contract-key".into()),
            api_base: Some(arguments.get_item("api_base")?.unwrap().extract()?),
            custom_llm_provider: Some("anthropic".into()),
            extra_headers: None,
            provider_specific_header: None,
            timeout: Some(Duration::from_secs(5)),
            shaping: MessagesShaping::default(),
        })
    }
    fn invoke(&mut self, _: Python<'_>, op: Infallible) -> Result<(), InvokeError<Error>> {
        match op {}
    }
    fn complete(&mut self, py: Python<'_>, response: MessagesOutput) -> PyResult<Py<PyAny>> {
        match response {
            MessagesOutput::Message(message) => to_py(py, &*message),
            MessagesOutput::Streamed => Ok(py.None()),
        }
    }
    fn head(&mut self, py: Python<'_>, _: MessagesStreamHead) -> PyResult<Py<PyAny>> {
        Ok(PyDict::new(py).into_any().unbind())
    }
    fn chunk(&mut self, py: Python<'_>, chunk: Bytes) -> PyResult<Py<PyAny>> {
        Ok(PyBytes::new(py, &chunk).into_any().unbind())
    }
    fn classify(&self, py: Python<'_>, _: Error) -> PyResult<PyErr> {
        Ok(PyErr::from_value(self.error.bind(py).clone()))
    }
    fn host_error(error: &PyErr) -> Error {
        Error::InvalidRequest(error.to_string())
    }
    fn close(&mut self, _: Python<'_>) {}
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.error)
    }
}

fn no_preflight(_py: Python<'_>, _arguments: &Bound<'_, PyDict>) -> PyResult<()> {
    Ok(())
}

#[pyclass]
struct Invoke {
    kwargs: Py<PyDict>,
    error: Py<PyAny>,
    asynchronous: bool,
}
#[pymethods]
impl Invoke {
    fn __call__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        run_legacy_call(
            py,
            LegacySurface {
                call_type: "anthropic_messages",
                input_description: "Messages",
                stream: Some(PassThroughStream {
                    url_route: "/v1/messages",
                    endpoint_type: "anthropic",
                }),
            },
            PublicCall::capture(
                py.None().bind(py),
                &PyTuple::empty(py),
                self.kwargs.bind(py),
            )?,
            messages_machine(
                &CoreResources::new(Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver)))),
                &Resolution::from(&HttpSettings::default()).config,
                Arc::new(NoSecrets),
            )
            .expect("default HTTP settings build a client"),
            MessagesHost {
                error: self.error.clone_ref(py),
            },
            no_preflight,
            self.asynchronous,
        )
    }
}

#[derive(Debug, FromPyObject)]
pub struct Event {
    pub name: String,
    pub value: Py<PyAny>,
    pub thread: u64,
    pub task: Option<u64>,
    pub context: String,
}

pub struct Case {
    pub state: Py<PyAny>,
    pub server: MockServer,
    mode: Mode,
    _guard: MutexGuard<'static, ()>,
}

impl Case {
    pub fn new(
        mode: Mode,
        upstream: Upstream,
        registration: &str,
        fault: Option<(&str, bool)>,
    ) -> Self {
        static SERIAL: Mutex<()> = Mutex::new(());
        let guard = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
        let server = runtime().block_on(MockServer::start());
        let template = match upstream {
            Upstream::Success => ResponseTemplate::new(200).set_body_json(response()),
            Upstream::Failure => ResponseTemplate::new(400).set_body_json(
                json!({"error": {"message": "provider failed", "type": "invalid_request_error"}}),
            ),
            Upstream::Stream => {
                ResponseTemplate::new(200).set_body_raw(stream_events(), "text/event-stream")
            }
        };
        runtime().block_on(
            Mock::given(method("POST"))
                .and(path("/v1/messages"))
                .respond_with(template)
                .mount(&server),
        );
        Python::initialize();
        let state =
            Python::attach(|py| {
                py.import("os")
                    .unwrap()
                    .getattr("environ")
                    .unwrap()
                    .set_item("LITELLM_LOCAL_MODEL_COST_MAP", "True")
                    .unwrap();
                py.import("sys")
                    .unwrap()
                    .getattr("path")
                    .unwrap()
                    .call_method1(
                        "insert",
                        (0, concat!(env!("CARGO_MANIFEST_DIR"), "/../../..")),
                    )
                    .unwrap();
                let module = PyModule::from_code(
                    py,
                    &CString::new(include_str!("fixtures.py")).unwrap(),
                    c"fixtures.py",
                    c"callback_contract_fixtures",
                )
                .unwrap();
                let kwargs = to_py(py, &json!({
                "model": MODEL, "custom_llm_provider": "anthropic", "api_key": "contract-key",
                "api_base": server.uri(), "messages": [{"role": "user", "content": "original"}],
                "max_tokens": 16, "stream": matches!(upstream, Upstream::Stream),
                "tools": [{"name": "original", "input_schema": {"type": "object"}}],
                "litellm_call_id": "callback-contract",
            })).unwrap();
                let (hook, cancellation) = fault.unwrap_or(("", false));
                module
                    .getattr("Scenario")
                    .unwrap()
                    .call1((
                        kwargs,
                        mode.asynchronous(),
                        registration,
                        hook,
                        cancellation,
                    ))
                    .unwrap()
                    .unbind()
            });
        Self {
            state,
            server,
            mode,
            _guard: guard,
        }
    }

    pub fn run(&self, finish: &str) {
        Python::attach(|py| {
            let state = self.state.bind(py);
            let invoke = Py::new(
                py,
                Invoke {
                    kwargs: state.getattr("kwargs").unwrap().extract().unwrap(),
                    error: state.getattr("provider_error").unwrap().unbind(),
                    asynchronous: self.mode.asynchronous(),
                },
            )
            .unwrap();
            state
                .call_method1("execute", (invoke, self.mode.asynchronous(), finish))
                .unwrap();
        });
    }

    pub fn events(&self) -> Vec<Event> {
        Python::attach(|py| {
            self.state
                .bind(py)
                .getattr("recorder")
                .unwrap()
                .getattr("events")
                .unwrap()
                .extract()
                .unwrap()
        })
    }

    pub fn names(&self) -> Vec<String> {
        self.events().into_iter().map(|event| event.name).collect()
    }

    pub fn event(&self, name: &str) -> Event {
        let matching: Vec<_> = self
            .events()
            .into_iter()
            .filter(|event| event.name == name)
            .collect();
        let [event]: [Event; 1] = matching
            .try_into()
            .unwrap_or_else(|events| panic!("expected one {name}, got {events:?}"));
        event
    }

    pub fn requests(&self) -> Vec<wiremock::Request> {
        runtime().block_on(self.server.received_requests()).unwrap()
    }

    pub fn assert_success(&self) {
        Python::attach(|py| {
            let error = self.state.bind(py).getattr("error").unwrap();
            assert!(
                error.is_none(),
                "unexpected error: {error}; callbacks: {:?}",
                self.names()
            );
        });
    }

    pub fn assert_error_is(&self, attribute: &str) {
        Python::attach(|py| {
            let state = self.state.bind(py);
            let expected = match attribute {
                "callback" => state
                    .getattr("recorder")
                    .unwrap()
                    .getattr("failure")
                    .unwrap(),
                _ => state.getattr(attribute).unwrap(),
            };
            let actual = state.getattr("error").unwrap();
            assert!(actual.is(&expected), "expected {expected}, got {actual}");
        });
    }
}

impl Drop for Case {
    fn drop(&mut self) {
        Python::attach(|py| {
            if let Err(error) = self.state.bind(py).call_method0("close") {
                error.print(py);
                if !std::thread::panicking() {
                    panic!("callback fixture cleanup failed: {error}");
                }
            }
        });
    }
}
