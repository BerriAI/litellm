use std::{
    ffi::CString,
    path::Path,
    sync::{Arc, Mutex, PoisonError},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, Activation, Callback, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, Patch,
    Selection, Subscription, Verdict, typed,
};
use litellm_callbacks_python::RustCallback;
use pyo3::{prelude::*, types::PyDict};
use serde_json::json;

#[derive(Default)]
struct Seen(Mutex<Option<String>>);

struct Example {
    manifest: Manifest,
    log: Arc<Mutex<Vec<String>>>,
}

impl Example {
    fn new() -> Self {
        Self {
            manifest: Manifest {
                selection: Selection {
                    activation: Activation::Required,
                    ..Selection::named("rust_example")
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
                has_request_state: true,
            },
            log: Arc::default(),
        }
    }

    fn pre_call(&self, input: &HookInput, state: Option<&Seen>) -> HookOutput {
        let model = input.fields.get::<typed::Model>()?.unwrap_or_default();
        if model == "blocked" {
            let mut abort = Abort::new(403, "blocked by the rust example");
            abort.code = Some("rust_block".into());
            abort.headers.insert("x-rust".into(), "blocked".into());
            return Ok(Some(Verdict::Abort(abort)));
        }
        if let Some(seen) = state {
            *seen.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(model.clone());
        }
        Ok(Some(Verdict::Patch(
            Patch::default().set(Field::Model, json!(format!("rust/{model}"))),
        )))
    }

    fn attempt_ended(&self, input: &HookInput, state: Option<&Seen>) -> HookOutput {
        let remembered = state.and_then(|seen| {
            seen.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        });
        let model = input.fields.get::<typed::Model>()?;
        let call_id = input.call.as_ref().map(|call| call.ids.call_id.clone());
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(format!(
                "attempt_ended {remembered:?} {model:?} {call_id:?}"
            ));
        Ok(None)
    }
}

impl Callback for Example {
    type State = Seen;

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn request_state(&self) -> Option<Seen> {
        Some(Seen::default())
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a Seen>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            match input.hook {
                Hook::PreCall => self.pre_call(input, state),
                Hook::AttemptEnded => self.attempt_ended(input, state),
                _ => Ok(None),
            }
        })
    }
}

#[rstest::rstest]
fn a_rust_callback_runs_at_its_list_position_between_two_legacy_entries() {
    let example = Arc::new(Example::new());
    let log = Arc::clone(&example.log);
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../..");
    let script = CString::new(include_str!("round_trip/script.py")).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("round_trip/manifest.json")).unwrap();
    assert_eq!(serde_json::to_value(example.manifest()).unwrap(), manifest);
    Python::initialize();
    let (seen, served_model, blocked) = Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item(
                "rust",
                Bound::new(py, RustCallback::new(example).unwrap()).unwrap(),
            )
            .unwrap();
        locals.set_item("repo", repo.to_str().unwrap()).unwrap();
        if let Err(error) = py.run(&script, Some(&locals), Some(&locals)) {
            error.display(py);
            panic!("the round trip script failed");
        }
        let item = |name: &str| locals.get_item(name).unwrap().unwrap();
        (
            item("seen").extract::<Vec<String>>().unwrap(),
            item("served_model").extract::<String>().unwrap(),
            item("blocked_abort")
                .extract::<(
                    u16,
                    String,
                    Option<String>,
                    std::collections::BTreeMap<String, String>,
                )>()
                .unwrap(),
        )
    });

    assert_eq!(
        seen,
        [
            "first pre_call client",
            "last pre_call rust/gpt",
            "first attempt_ended rust/gpt",
            "last attempt_ended rust/gpt",
            "first pre_call client",
        ]
    );
    assert_eq!(served_model, "rust/gpt");
    assert_eq!(
        blocked,
        (
            403,
            "blocked by the rust example".to_owned(),
            Some("rust_block".to_owned()),
            [("x-rust".to_owned(), "blocked".to_owned())].into(),
        )
    );
    assert_eq!(
        *log.lock().unwrap_or_else(PoisonError::into_inner),
        [r#"attempt_ended Some("gpt") Some("rust/gpt") Some("call-1")"#]
    );
}

struct Panicking(Manifest);

impl Callback for Panicking {
    type State = ();

    fn manifest(&self) -> &Manifest {
        &self.0
    }

    fn request_state(&self) -> Option<()> {
        panic!("no state for this request")
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        _state: Option<&'a ()>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            let usage = input.fields.raw(Field::Usage).ok().flatten().cloned();
            panic!("usage was {usage:?}")
        })
    }
}

#[rstest::rstest]
fn a_panicking_rust_callback_raises_a_runtime_error_python_can_contain() {
    let panicking = Arc::new(Panicking(Manifest {
        selection: Selection::named("rust_panicking"),
        subscriptions: vec![Subscription {
            reads: FieldSet::of(&[Field::Usage]),
            ..Subscription::on(Hook::AttemptEnded)
        }],
        has_request_state: true,
    }));
    let script = CString::new(
        r#"
import asyncio

def raised(run):
    try:
        run()
    except BaseException as error:
        return (type(error).__name__, isinstance(error, Exception), str(error))
    return None

async def handle():
    return await rust.handle("attempt_ended", {}, None, None)

handled = raised(lambda: asyncio.run(handle()))
state = raised(rust.request_state)
"#,
    )
    .unwrap();
    Python::initialize();
    let (handled, state) = Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item(
                "rust",
                Bound::new(py, RustCallback::new(panicking).unwrap()).unwrap(),
            )
            .unwrap();
        if let Err(error) = py.run(&script, Some(&locals), Some(&locals)) {
            error.display(py);
            panic!("the panic script failed");
        }
        let item = |name: &str| {
            locals
                .get_item(name)
                .unwrap()
                .unwrap()
                .extract::<(String, bool, String)>()
                .unwrap()
        };
        (item("handled"), item("state"))
    });

    assert_eq!(
        handled,
        (
            "RuntimeError".to_owned(),
            true,
            "the rust callback panicked: usage was None".to_owned()
        )
    );
    assert_eq!(
        state,
        (
            "RuntimeError".to_owned(),
            true,
            "the rust callback panicked: no state for this request".to_owned()
        )
    );
}
