//! End to end: a real `CallSession` driven through a real Python worker. The callback file
//! below is an ordinary SDK callback; nothing in it knows it runs out of process.
//!
//! Ignored by default because it needs an interpreter that can import `litellm`:
//!
//!     PYO3_PYTHON=/path/to/.venv/bin/python cargo test -p litellm-callbacks-v1-exthost -- --ignored

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use litellm_callbacks_v1::{CallFacts, CallSession, ErrorFacts, PatchError};
use litellm_callbacks_v1_exthost::{ExtHostError, Report, Worker};
use litellm_host::event::{FailureOrigin, RequestContext, Timing, WireRequest};
use serde_json::{Map, Value, json};

const CALLBACKS: &str = r#"
import json
import os

from litellm.callbacks_v1 import before_send, on_event

LOG = os.environ["CALLBACK_LOG"]


@on_event("call.started", "request.sending", "response.received", "call.succeeded", name="recorder")
async def record(event):
    with open(LOG, "a") as log:
        log.write(json.dumps([event["seq"], event["event"]["type"], event["event"].get("body")]) + "\n")


@on_event("call.started", name="flaky")
def flaky(event):
    raise ValueError("boom")


@before_send
def tag(request):
    return {"headers": {"set": [["X-Team", "core"]]}, "body": {**request["body"], "tagged": True}}


@before_send
async def guard(request):
    if request["body"].get("input") == "steal":
        return {"headers": {"set": [["Authorization", "stolen"]]}}
    if request["body"].get("input") == "refuse":
        raise PermissionError("not this input")
    return None
"#;

const TIMING: Timing = Timing {
    start_time: 10.0,
    end_time: 12.5,
};

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = env::temp_dir().join(format!("exthost-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(path.join("modules")).unwrap();
        fs::create_dir_all(path.join("cwd")).unwrap();
        fs::write(path.join("modules/gateway_callbacks.py"), CALLBACKS).unwrap();
        Self(path)
    }

    fn log(&self) -> Vec<Value> {
        fs::read_to_string(self.0.join("log"))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The worker gets a cleared environment and an empty working directory: it sees what it
/// is given and nothing of the gateway's.
fn worker(scratch: &Scratch) -> Worker {
    let python = env::var("PYO3_PYTHON").expect("PYO3_PYTHON names an interpreter with litellm");
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let python_path = env::join_paths([repository, scratch.0.join("modules")]).unwrap();
    let mut command = Command::new(python);
    command
        .args([
            "-m",
            "litellm.callbacks_v1.host",
            "--load",
            "gateway_callbacks",
        ])
        .env_clear()
        .env("PYTHONPATH", python_path)
        .env("LITELLM_MODE", "PRODUCTION")
        .env("CALLBACK_LOG", scratch.0.join("log"))
        .current_dir(scratch.0.join("cwd"));
    Worker::spawn(command, &scratch.0.join("socket")).unwrap()
}

fn wire(input: &str) -> WireRequest {
    WireRequest {
        url: "https://provider.example/v1".to_string(),
        headers: vec![("Authorization".to_string(), "secret".to_string())],
        body: json!({"input": input}),
    }
}

fn context() -> RequestContext {
    RequestContext {
        model: "model".to_string(),
        custom_llm_provider: "provider".to_string(),
        optional_params: json!({}),
        secret_fields: Vec::new(),
        api_key: None,
    }
}

fn start(worker: &mut Worker) -> CallSession {
    let facts = CallFacts {
        start_time: TIMING.start_time,
        asynchronous: true,
        metadata: Map::new(),
        metadata_dropped: Vec::new(),
    };
    let (session, started) = CallSession::start(
        "call-123".to_string(),
        "ocr",
        worker.subscriptions().to_vec(),
        facts,
    );
    worker.observe(&started).unwrap();
    session
}

#[test]
#[ignore = "needs PYO3_PYTHON to name an interpreter that can import litellm"]
fn a_call_reaches_python_callbacks_and_their_patches_reach_the_wire() {
    let scratch = Scratch::new("call");
    let mut worker = worker(&scratch);
    let names: Vec<&str> = worker
        .subscriptions()
        .iter()
        .map(|subscription| subscription.name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "recorder",
            "flaky",
            "gateway_callbacks.tag",
            "gateway_callbacks.guard"
        ]
    );

    let mut session = start(&mut worker);
    let interception = worker
        .intercept(session.interception(wire("hello"), context()))
        .unwrap();
    let (sending, sent) = session.request_sending(interception);
    worker.observe(&sending).unwrap();
    worker
        .observe(&session.response_received("{}".to_string()))
        .unwrap();
    worker
        .observe(&session.succeeded(TIMING, json!({"id": "response"}), None))
        .unwrap();

    assert_eq!(sent.body, json!({"input": "hello", "tagged": true}));
    assert!(
        sent.headers
            .contains(&("X-Team".to_string(), "core".to_string()))
    );
    assert!(
        sent.headers
            .contains(&("Authorization".to_string(), "secret".to_string()))
    );
    assert_eq!(
        worker.flush().unwrap(),
        vec![Report {
            subscriber: "flaky".to_string(),
            event: Some("call.started".to_string()),
            message: "boom".to_string(),
        }]
    );
    assert_eq!(
        scratch.log(),
        vec![
            json!([0, "call.started", null]),
            json!([1, "request.sending", {"input": "hello", "tagged": true}]),
            json!([2, "response.received", "{}"]),
            json!([3, "call.succeeded", null]),
        ]
    );
}

#[test]
#[ignore = "needs PYO3_PYTHON to name an interpreter that can import litellm"]
fn a_worker_cannot_patch_a_credential_and_its_refusal_is_the_gateways_to_act_on() {
    let scratch = Scratch::new("refusals");
    let mut worker = worker(&scratch);

    let session = start(&mut worker);
    let stolen = worker
        .intercept(session.interception(wire("steal"), context()))
        .unwrap_err();
    assert!(matches!(
        stolen,
        ExtHostError::Patch { ref subscriber, error: PatchError::ProtectedHeader(_) }
            if subscriber == "gateway_callbacks.guard"
    ));

    let refused = worker
        .intercept(session.interception(wire("refuse"), context()))
        .unwrap_err();
    assert!(matches!(
        refused,
        ExtHostError::Interceptor { ref error_class, ref message, .. }
            if error_class == "builtins.PermissionError" && message == "not this input"
    ));

    let failed = session.failed(
        TIMING,
        FailureOrigin::Call,
        ErrorFacts {
            class: "builtins.PermissionError".to_string(),
            message: "not this input".to_string(),
            status_code: None,
        },
    );
    assert!(failed.observers.is_empty());
    worker.flush().unwrap();
}
