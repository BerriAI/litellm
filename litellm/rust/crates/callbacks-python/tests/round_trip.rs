use std::{
    ffi::CString,
    path::Path,
    sync::{Arc, Mutex, PoisonError},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, Activation, Callback, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, Patch,
    RequestState, Selection, Subscription, Verdict, typed,
};
use litellm_callbacks_python::NativeCallback;
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
                    ..Selection::named("native_example")
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

    fn pre_call(&self, input: &HookInput) -> HookOutput {
        let model = input.fields.get::<typed::Model>()?.unwrap_or_default();
        if model == "blocked" {
            let mut abort = Abort::new(403, "blocked by the native example");
            abort.code = Some("native_block".into());
            abort.headers.insert("x-native".into(), "blocked".into());
            return Ok(Some(Verdict::Abort(abort)));
        }
        if let Some(seen) = input.state::<Seen>() {
            *seen.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(model.clone());
        }
        Ok(Some(Verdict::Patch(
            Patch::default().set(Field::Model, json!(format!("native/{model}"))),
        )))
    }

    fn attempt_ended(&self, input: &HookInput) -> HookOutput {
        let remembered = input.state::<Seen>().and_then(|seen| {
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
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn request_state(&self) -> Option<RequestState> {
        Some(Arc::new(Seen::default()))
    }

    fn handle<'a>(&'a self, input: &'a HookInput) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            match input.hook {
                Hook::PreCall => self.pre_call(input),
                Hook::AttemptEnded => self.attempt_ended(input),
                _ => Ok(None),
            }
        })
    }
}

#[rstest::rstest]
fn a_native_callback_runs_at_its_list_position_between_two_legacy_entries() {
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
                "native",
                Bound::new(py, NativeCallback::new(example).unwrap()).unwrap(),
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
            "last pre_call native/gpt",
            "first attempt_ended native/gpt",
            "last attempt_ended native/gpt",
            "first pre_call client",
        ]
    );
    assert_eq!(served_model, "native/gpt");
    assert_eq!(
        blocked,
        (
            403,
            "blocked by the native example".to_owned(),
            Some("native_block".to_owned()),
            [("x-native".to_owned(), "blocked".to_owned())].into(),
        )
    );
    assert_eq!(
        *log.lock().unwrap_or_else(PoisonError::into_inner),
        [r#"attempt_ended Some("gpt") Some("native/gpt") Some("call-1")"#]
    );
}
