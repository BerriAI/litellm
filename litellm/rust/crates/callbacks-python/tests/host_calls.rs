use std::{
    ffi::CString,
    path::Path,
    sync::{Mutex, PoisonError},
    time::Duration,
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Activation, Callback, CallbackError, Field, FieldSet, Hook, HookInput, HookOutput, Host,
    HostCall, HostCallError, Manifest, Patch, Selection, Subscription, Verdict, typed,
};
use litellm_callbacks_python::RustCallback;
use pyo3::{prelude::*, types::PyDict};
use rstest::rstest;
use serde::Serialize;
use serde_json::json;

static DROPPED: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Guard(String);

impl Drop for Guard {
    fn drop(&mut self) {
        DROPPED
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(self.0.clone());
    }
}

#[pyfunction]
fn dropped(scenario: &str) -> bool {
    DROPPED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .any(|name| name == scenario)
}

#[derive(Serialize)]
struct Lookup {
    scenario: String,
    step: u32,
}

impl HostCall for Lookup {
    const NAME: &'static str = "lookup";
    type Reply = String;
}

struct Scripted(Manifest);

fn patched(model: String) -> HookOutput {
    Ok(Some(Verdict::Patch(
        Patch::default().set(Field::Model, json!(model)),
    )))
}

async fn lookup(host: &Host, scenario: &str, step: u32) -> Result<String, HostCallError> {
    host.call(Lookup {
        scenario: scenario.to_owned(),
        step,
    })
    .await
}

impl Scripted {
    fn new() -> Self {
        Self(Manifest {
            selection: Selection {
                activation: Activation::Required,
                ..Selection::named("rust_host_calls")
            },
            subscriptions: vec![
                Subscription {
                    reads: FieldSet::of(&[Field::Model]),
                    writes: FieldSet::of(&[Field::Model]),
                    may_abort: true,
                    ..Subscription::on(Hook::PreCall)
                },
                Subscription {
                    reads: FieldSet::of(&[Field::Model]),
                    ..Subscription::on(Hook::AttemptEnded)
                },
            ],
            has_request_state: false,
        })
    }

    async fn run(&self, input: &HookInput, host: Host) -> HookOutput {
        let scenario = input.fields.get::<typed::Model>()?.unwrap_or_default();
        match scenario.as_str() {
            "none" => patched("rust/none".to_owned()),
            "sequential" => {
                let first = lookup(&host, &scenario, 1).await?;
                tokio::time::sleep(Duration::from_millis(30)).await;
                let second = lookup(&host, &scenario, 2).await?;
                patched(format!("{first}+{second}"))
            }
            "handled" => match lookup(&host, &scenario, 1).await {
                Err(HostCallError::Raised { exception, .. }) => {
                    patched(format!("handled {}", exception.message()))
                }
                other => Err(CallbackError::Failed(format!("{other:?}"))),
            },
            "cancel_rust" => {
                let _guard = Guard(scenario.clone());
                lookup(&host, &scenario, 1).await?;
                tokio::time::sleep(Duration::from_secs(60)).await;
                lookup(&host, &scenario, 2).await?;
                Ok(None)
            }
            "cancel_host" => {
                let _guard = Guard(scenario.clone());
                lookup(&host, &scenario, 1).await?;
                Ok(None)
            }
            "panic" => panic!("host calls panicked"),
            _ => patched(lookup(&host, &scenario, 1).await?),
        }
    }
}

impl Callback for Scripted {
    type State = ();

    fn manifest(&self) -> &Manifest {
        &self.0
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a ()>,
    ) -> BoxFuture<'a, HookOutput> {
        self.handle_hosted(input, state, Host::detached())
    }

    fn handle_hosted<'a>(
        &'a self,
        input: &'a HookInput,
        _state: Option<&'a ()>,
        host: Host,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(self.run(input, host))
    }
}

#[rstest]
#[case::a_plain_value("plain_value")]
#[case::a_coroutine("coroutine")]
#[case::a_context_var_the_caller_set("reads_context_var")]
#[case::a_context_var_the_caller_sees_written("writes_context_var")]
#[case::two_calls_around_a_rust_await("sequential_calls")]
#[case::the_callers_task("same_task")]
#[case::the_request_scope("request_scope")]
#[case::the_original_exception("original_exception")]
#[case::an_exception_the_handler_handles("handled_exception")]
#[case::a_panic("panic_contained")]
#[case::cancelled_while_python_is_awaited("cancelled_in_host_call")]
#[case::cancelled_while_rust_is_pending("cancelled_in_rust")]
#[case::no_host_call("no_host_call")]
#[case::no_host("detached")]
#[case::a_sync_line("sync_line")]
#[case::off_path_without_a_loop("off_path_without_a_loop")]
#[case::a_sync_driver("sync_driver")]
fn a_rust_handler_awaits_python_host_calls(#[case] case: &str) {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../..");
    let script = CString::new(include_str!("host_calls/script.py")).unwrap();
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item(
                "rust",
                Bound::new(
                    py,
                    RustCallback::new(std::sync::Arc::new(Scripted::new())).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        locals.set_item("repo", repo.to_str().unwrap()).unwrap();
        locals
            .set_item("dropped", wrap_pyfunction!(dropped, py).unwrap())
            .unwrap();
        if let Err(error) = py.run(&script, Some(&locals), Some(&locals)) {
            error.display(py);
            panic!("the host call script failed to load");
        }
        if let Err(error) = locals.get_item(case).unwrap().unwrap().call0() {
            error.display(py);
            panic!("{case} failed");
        }
    });
}
