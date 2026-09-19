use std::collections::BTreeSet;

use litellm_auth::SecretValue;
use litellm_host::event::{MachineEvent, RawResponse, RequestContext, Timing, WireRequest};
use litellm_host_python::{LifecycleEvent, LifecycleStep, PythonLifecycle, from_py};
use pyo3::exceptions::asyncio::CancelledError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Value, json};

use crate::adapter::NextLifecycle;
use crate::call::NextSurface;
use crate::envelope::{EventKind, SCHEMA_V1};
use crate::registry::{Handler, Subscriber};
use crate::test_support::{local, namespace};

const TIMING: Timing = Timing {
    start_time: 10.0,
    end_time: 12.5,
};

fn subscriber(
    locals: &Bound<'_, PyDict>,
    name: &str,
    events: &[EventKind],
    observe: Option<(&str, bool)>,
    intercept: Option<(&str, bool)>,
) -> Subscriber {
    let handler = |spec: Option<(&str, bool)>| {
        spec.map(|(local_name, asynchronous)| {
            let callable = local(locals, local_name).unbind();
            if asynchronous {
                Handler::Async(callable)
            } else {
                Handler::Sync(callable)
            }
        })
    };
    Subscriber {
        name: name.to_string(),
        schema: SCHEMA_V1,
        events: events.iter().copied().collect::<BTreeSet<_>>(),
        observe: handler(observe),
        intercept: handler(intercept),
    }
}

fn lifecycle(subscribers: Vec<Subscriber>, asynchronous: bool) -> NextLifecycle {
    NextLifecycle::new(NextSurface { call_type: "ocr" }, subscribers, asynchronous)
}

fn started(adapter: &mut NextLifecycle, py: Python<'_>) {
    assert!(matches!(
        adapter
            .emit(py, LifecycleEvent::Started { start_time: 10.0 })
            .unwrap(),
        LifecycleStep::Done
    ));
}

fn context(secret: &str) -> RequestContext {
    RequestContext {
        model: "model".to_string(),
        custom_llm_provider: "provider".to_string(),
        optional_params: json!({"api_key": secret, "temperature": 0.2}),
        secret_fields: vec!["api_key".to_string()],
        api_key: Some(SecretValue::new(secret.to_string())),
    }
}

fn wire(secret: &str) -> WireRequest {
    WireRequest {
        url: "https://provider.example/v1".to_string(),
        headers: vec![
            ("Authorization".to_string(), secret.to_string()),
            ("X-Trace".to_string(), "one".to_string()),
        ],
        body: json!({"input": "hello"}),
    }
}

#[test]
fn full_sync_success_preserves_identity_sequence_and_secrets() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
import copy
seen = []
def observe(event):
    seen.append(copy.deepcopy(event))
response = {'id': 'response'}
",
        );
        let events = [
            EventKind::CallStarted,
            EventKind::RequestSending,
            EventKind::ResponseReceived,
            EventKind::CallSucceeded,
        ];
        let observer = subscriber(
            &locals,
            "sync-success",
            &events,
            Some(("observe", false)),
            None,
        );
        let mut adapter = lifecycle(vec![observer], false);
        started(&mut adapter, py);
        let arguments = PyDict::new(py);
        arguments.set_item("litellm_call_id", "call-123").unwrap();
        let LifecycleStep::Arguments(returned) =
            adapter.begin(py, arguments.clone().unbind(), 10.0).unwrap()
        else {
            panic!("begin did not return arguments");
        };
        assert!(returned.bind(py).is(&arguments));

        let secret = "sentinel-secret-78e4";
        let LifecycleStep::Wire(sent) = adapter
            .before_send(py, Box::new(wire(secret)), &context(secret))
            .unwrap()
        else {
            panic!("before_send did not return wire");
        };
        assert_eq!(sent.headers[0].1, secret);
        let raw = MachineEvent::ResponseReceived {
            raw: RawResponse {
                body: "raw response".to_string(),
            },
        };
        assert!(matches!(
            adapter.emit(py, LifecycleEvent::Machine(&raw)).unwrap(),
            LifecycleStep::Done
        ));
        let response = local(&locals, "response").unbind();
        let LifecycleStep::Response(returned) = adapter
            .after_success(py, response.clone_ref(py), TIMING)
            .unwrap()
        else {
            panic!("after_success did not return response");
        };
        assert!(returned.bind(py).is(response.bind(py)));
        assert!(matches!(
            adapter
                .emit(
                    py,
                    LifecycleEvent::Succeeded {
                        timing: TIMING,
                        response: &response,
                    },
                )
                .unwrap(),
            LifecycleStep::Done
        ));

        let seen = from_py::<Vec<Value>>(&local(&locals, "seen")).unwrap();
        assert_eq!(
            seen.iter()
                .map(|envelope| envelope["event"]["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "call.started",
                "request.sending",
                "response.received",
                "call.succeeded",
            ]
        );
        assert_eq!(
            seen.iter()
                .map(|envelope| envelope["seq"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert!(!serde_json::to_string(&seen).unwrap().contains(secret));
    });
}

#[test]
fn every_observer_gets_a_fresh_envelope() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
import copy
seen = []
def mutate(event):
    event['event']['body']['input'] = 'changed'
def record(event):
    seen.append(copy.deepcopy(event))
",
        );
        let event = [EventKind::RequestSending];
        let subscribers = vec![
            subscriber(&locals, "mutator", &event, Some(("mutate", false)), None),
            subscriber(&locals, "recorder", &event, Some(("record", false)), None),
        ];
        let mut adapter = lifecycle(subscribers, false);
        started(&mut adapter, py);
        adapter.begin(py, PyDict::new(py).unbind(), 10.0).unwrap();
        let LifecycleStep::Wire(sent) = adapter
            .before_send(py, Box::new(wire("secret")), &context("secret"))
            .unwrap()
        else {
            panic!("before_send did not return wire");
        };
        assert_eq!(sent.body, json!({"input": "hello"}));
        let seen = from_py::<Vec<Value>>(&local(&locals, "seen")).unwrap();
        assert_eq!(seen[0]["event"]["body"], json!({"input": "hello"}));
    });
}

#[test]
fn ordinary_observer_errors_are_reported_and_swallowed() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
seen = []
def fail(event):
    raise ValueError('observer failed')
def record(event):
    seen.append(event['event']['type'])
",
        );
        let event = [EventKind::CallStarted];
        let subscribers = vec![
            subscriber(
                &locals,
                "observer-error-401",
                &event,
                Some(("fail", false)),
                None,
            ),
            subscriber(
                &locals,
                "after-error",
                &event,
                Some(("record", false)),
                None,
            ),
        ];
        let mut adapter = lifecycle(subscribers, false);
        started(&mut adapter, py);
        assert!(matches!(
            adapter.begin(py, PyDict::new(py).unbind(), 10.0).unwrap(),
            LifecycleStep::Arguments(_)
        ));
        assert_eq!(
            local(&locals, "seen").extract::<Vec<String>>().unwrap(),
            ["call.started"]
        );
        let reports = py
            .import("litellm.rust_bridge.callbacks_next")
            .unwrap()
            .getattr("reports")
            .unwrap();
        let names = reports
            .try_iter()
            .unwrap()
            .filter_map(Result::ok)
            .filter_map(|report| report.get_item(0).ok()?.extract::<String>().ok())
            .collect::<Vec<_>>();
        assert!(names.contains(&"observer-error-401".to_string()));
    });
}

#[test]
fn cancellation_stops_observer_fanout() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
import asyncio
seen = []
def cancel(event):
    raise asyncio.CancelledError()
def record(event):
    seen.append(event)
",
        );
        let event = [EventKind::CallStarted];
        let subscribers = vec![
            subscriber(&locals, "cancel", &event, Some(("cancel", false)), None),
            subscriber(
                &locals,
                "after-cancel",
                &event,
                Some(("record", false)),
                None,
            ),
        ];
        let mut adapter = lifecycle(subscribers, false);
        started(&mut adapter, py);
        let error = match adapter.begin(py, PyDict::new(py).unbind(), 10.0) {
            Err(error) => error,
            Ok(_) => panic!("cancellation was swallowed"),
        };
        assert!(error.is_instance_of::<CancelledError>(py));
        assert_eq!(local(&locals, "seen").len().unwrap(), 0);
    });
}

#[test]
fn async_observers_resume_in_order_and_return_arguments() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
calls = []
def first(event):
    calls.append('first')
    return object()
def second(event):
    calls.append('second')
    return object()
",
        );
        let event = [EventKind::CallStarted];
        let subscribers = vec![
            subscriber(&locals, "async-first", &event, Some(("first", true)), None),
            subscriber(
                &locals,
                "async-second",
                &event,
                Some(("second", true)),
                None,
            ),
        ];
        let mut adapter = lifecycle(subscribers, true);
        started(&mut adapter, py);
        let arguments = PyDict::new(py).unbind();
        assert!(matches!(
            adapter.begin(py, arguments.clone_ref(py), 10.0).unwrap(),
            LifecycleStep::Await(_)
        ));
        assert!(matches!(
            adapter.resume(py, Ok(py.None())).unwrap(),
            LifecycleStep::Await(_)
        ));
        let LifecycleStep::Arguments(returned) = adapter.resume(py, Ok(py.None())).unwrap() else {
            panic!("resume did not return arguments");
        };
        assert!(returned.bind(py).is(arguments.bind(py)));
        assert_eq!(
            local(&locals, "calls").extract::<Vec<String>>().unwrap(),
            ["first", "second"]
        );
    });
}

#[test]
fn interceptors_chain_and_observers_see_the_final_wire() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
seen = []
def first(request):
    return {'headers': {'set': [['X-Added', 'yes']]}, 'body': {'step': 1}}
def second(request):
    seen.append(request)
    return None
def observe(event):
    seen.append(event)
",
        );
        let subscribers = vec![
            subscriber(&locals, "first", &[], None, Some(("first", false))),
            subscriber(&locals, "second", &[], None, Some(("second", false))),
            subscriber(
                &locals,
                "observer",
                &[EventKind::RequestSending],
                Some(("observe", false)),
                None,
            ),
        ];
        let mut adapter = lifecycle(subscribers, false);
        started(&mut adapter, py);
        adapter.begin(py, PyDict::new(py).unbind(), 10.0).unwrap();
        let LifecycleStep::Wire(sent) = adapter
            .before_send(py, Box::new(wire("secret")), &context("secret"))
            .unwrap()
        else {
            panic!("before_send did not return wire");
        };
        assert_eq!(sent.body, json!({"step": 1}));
        assert_eq!(
            sent.headers.last().unwrap(),
            &("X-Added".to_string(), "yes".to_string())
        );
        let seen = from_py::<Vec<Value>>(&local(&locals, "seen")).unwrap();
        assert_eq!(seen[0]["body"], json!({"step": 1}));
        assert_eq!(seen[1]["event"]["body"], json!({"step": 1}));
    });
}

#[test]
fn invalid_interceptor_patch_stops_later_handlers() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
seen = []
def invalid(request):
    return {'headers': {'set': [['Authorization', 'replacement']]}}
def later(request):
    seen.append('later')
def observe(event):
    seen.append('observe')
",
        );
        let subscribers = vec![
            subscriber(
                &locals,
                "invalid-patch",
                &[],
                None,
                Some(("invalid", false)),
            ),
            subscriber(&locals, "later", &[], None, Some(("later", false))),
            subscriber(
                &locals,
                "observer",
                &[EventKind::RequestSending],
                Some(("observe", false)),
                None,
            ),
        ];
        let mut adapter = lifecycle(subscribers, false);
        started(&mut adapter, py);
        adapter.begin(py, PyDict::new(py).unbind(), 10.0).unwrap();
        let error = match adapter.before_send(py, Box::new(wire("secret")), &context("secret")) {
            Err(error) => error,
            Ok(_) => panic!("invalid patch was accepted"),
        };
        assert!(error.to_string().contains("invalid-patch"));
        assert_eq!(local(&locals, "seen").len().unwrap(), 0);
    });
}

#[test]
fn projection_failure_is_observed_without_failing_terminal_dispatch() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
seen = []
def observe(event):
    seen.append(event)
class Broken:
    project_error = True
response = Broken()
",
        );
        let observer = subscriber(
            &locals,
            "terminal",
            &[EventKind::CallSucceeded],
            Some(("observe", false)),
            None,
        );
        let mut adapter = lifecycle(vec![observer], false);
        started(&mut adapter, py);
        adapter.begin(py, PyDict::new(py).unbind(), 10.0).unwrap();
        adapter.opened(py).unwrap();
        let response = local(&locals, "response").unbind();
        assert!(matches!(
            adapter
                .emit(
                    py,
                    LifecycleEvent::Succeeded {
                        timing: TIMING,
                        response: &response,
                    },
                )
                .unwrap(),
            LifecycleStep::Done
        ));
        let seen = from_py::<Vec<Value>>(&local(&locals, "seen")).unwrap();
        assert_eq!(seen[0]["event"]["streamed"], true);
        assert!(seen[0]["event"]["response"].is_null());
        assert!(
            seen[0]["event"]["response_error"]
                .as_str()
                .unwrap()
                .contains("projection failed")
        );
    });
}

#[test]
fn resume_without_pending_fails_and_close_is_idempotent() {
    Python::initialize();
    Python::attach(|py| {
        let _locals = namespace(py, c"");
        let mut adapter = lifecycle(Vec::new(), false);
        assert!(adapter.resume(py, Ok(py.None())).is_err());
        adapter.close(py);
        adapter.close(py);
    });
}
