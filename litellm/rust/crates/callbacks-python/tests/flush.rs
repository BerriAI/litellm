use std::{
    ffi::CString,
    path::Path,
    sync::{Arc, Mutex, PoisonError},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Callback, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, Selection, Subscription,
    typed,
};
use litellm_callbacks_python::RustCallback;
use pyo3::{prelude::*, types::PyDict};

#[derive(Default)]
struct Rows {
    queued: Vec<String>,
    written: Vec<String>,
    flushes: usize,
}

struct Queueing {
    manifest: Manifest,
    rows: Arc<Mutex<Rows>>,
    broken: bool,
}

impl Queueing {
    fn new(name: &str, broken: bool) -> Self {
        Self {
            manifest: Manifest {
                selection: Selection::named(name),
                subscriptions: vec![Subscription {
                    reads: FieldSet::of(&[Field::Model]),
                    awaited: true,
                    ..Subscription::on(Hook::AttemptEnded)
                }],
                has_request_state: false,
            },
            rows: Arc::default(),
            broken,
        }
    }

    fn rows(&self) -> std::sync::MutexGuard<'_, Rows> {
        self.rows.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Callback for Queueing {
    type State = ();

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        _state: Option<&'a ()>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            let model = input.fields.get::<typed::Model>()?.unwrap_or_default();
            self.rows().queued.push(model);
            Ok(None)
        })
    }

    fn flush(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            tokio::task::yield_now().await;
            assert!(!self.broken, "the store is gone");
            let mut rows = self.rows();
            rows.flushes += 1;
            let queued = std::mem::take(&mut rows.queued);
            rows.written.extend(queued);
        })
    }
}

const SCRIPT: &str = r#"
import asyncio
import logging
import sys
import types
from pathlib import Path

repo = Path(globals()["repo"])
if "litellm" not in sys.modules:
    stub = types.ModuleType("litellm")
    stub.__path__ = [str(repo / "litellm")]
    sys.modules["litellm"] = stub

from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enforcement import FieldView
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.rust import flush_rust_loggers, rust_logger

errors = []


class Collect(logging.Handler):
    def emit(self, record):
        errors.append(record.getMessage())


logging.getLogger("LiteLLM").addHandler(Collect())

queueing = rust_logger(globals()["queueing"])
broken = rust_logger(globals()["broken"])


async def main():
    for model in ("gpt-a", "gpt-b"):
        view = FieldView({Field.model: model})
        await queueing.on_attempt_ended(view, CallContext(hook=Hook.attempt_ended, call=None, state=None))
    await flush_rust_loggers(["s3_v2", broken, queueing, queueing, object()])


asyncio.run(main())
"#;

#[rstest::rstest]
fn shutdown_flush_writes_every_queued_row_once_even_when_another_rust_callback_fails() {
    let queueing = Arc::new(Queueing::new("queueing", false));
    let broken = Arc::new(Queueing::new("broken", true));
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../..");
    Python::initialize();
    let errors = Python::attach(|py| {
        let locals = PyDict::new(py);
        let callback = |port: &Arc<Queueing>| {
            Bound::new(py, RustCallback::new(Arc::clone(port) as _).unwrap()).unwrap()
        };
        locals.set_item("queueing", callback(&queueing)).unwrap();
        locals.set_item("broken", callback(&broken)).unwrap();
        locals.set_item("repo", repo.to_str().unwrap()).unwrap();
        if let Err(error) = py.run(&CString::new(SCRIPT).unwrap(), Some(&locals), Some(&locals)) {
            error.display(py);
            panic!("the flush script failed");
        }
        locals
            .get_item("errors")
            .unwrap()
            .unwrap()
            .extract::<Vec<String>>()
            .unwrap()
    });

    let rows = queueing.rows();
    assert_eq!(rows.written, ["gpt-a", "gpt-b"]);
    assert!(rows.queued.is_empty());
    assert_eq!(rows.flushes, 1);
    assert_eq!(broken.rows().flushes, 0);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].starts_with("flushing rust callback broken failed"),
        "{errors:?}"
    );
    assert!(errors[0].contains("the store is gone"), "{errors:?}");
}
