use std::{
    ffi::CString,
    path::Path,
    sync::{Arc, Mutex, PoisonError},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Callback, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, PayloadRef, RenderOptions,
    Selection, Subscription,
};
use litellm_callbacks_python::{RustCallback, payload};
use pyo3::{prelude::*, types::PyDict};
use rstest::{fixture, rstest};

#[derive(Default)]
struct Kept {
    at_hook: Vec<Result<Vec<u8>, String>>,
    payloads: Vec<Arc<PayloadRef>>,
}

struct Keeper {
    manifest: Manifest,
    kept: Arc<Mutex<Kept>>,
}

impl Callback for Keeper {
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
            let Some(payload) = input.fields.payload()? else {
                return Ok(None);
            };
            let rendered = payload.render(RenderOptions::default()).await;
            let mut kept = self.kept.lock().unwrap_or_else(PoisonError::into_inner);
            kept.at_hook.push(rendered);
            kept.payloads.push(payload);
            Ok(None)
        })
    }
}

const STUB: &str = r#"
import sys
import types
from pathlib import Path

repo = Path(globals()["repo"])
if "litellm" not in sys.modules:
    stub = types.ModuleType("litellm")
    stub.__path__ = [str(repo / "litellm")]
    sys.modules["litellm"] = stub
"#;

const SCRIPT: &str = r#"
import asyncio
from pathlib import Path

from pydantic import BaseModel

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


class Tagged(BaseModel):
    name: str
    weights: tuple[float, ...]


deep = "bottom"
for _ in range(DEFAULT_MAX_RECURSE_DEPTH + 3):
    deep = [deep]
nested = {"seen": ["first"], 7: "an int key", ("a", "b"): "a tuple key"}
nested["self"] = nested
payload = {
    "id": "log-1",
    "metadata": nested,
    "tags": {"b", "a", "c"},
    "big": 2**80,
    "negative_big": -(2**70),
    "ratio": float("nan"),
    "infinite": float("inf"),
    "text": "nul\x00byte é 😀 \"quoted\"\n",
    "tuple": (1, 2.5, None, True),
    "model": Tagged(name="m", weights=(0.1, 1e-7, 1e22)),
    "deep": deep,
    "object": Path("/tmp/a"),
    "messages": [{"role": "user", "content": "secret prompt"}],
}
copy = dict(payload)


async def handle():
    return await rust.handle("attempt_ended", {"standard_logging_payload_ref": copy}, None, None)


asyncio.run(handle())
at_hook = safe_dumps(copy)
nested["seen"].append("after the hook")
nested["added"] = {"later": 1}
payload["id"] = "changed on the original, not the copy"
later = safe_dumps(copy)
prompts_only = safe_dumps({**copy, "messages": None, "response": None})
"#;

#[fixture]
fn keeper() -> Arc<Keeper> {
    Arc::new(Keeper {
        manifest: Manifest {
            selection: Selection::named("payload_keeper"),
            subscriptions: vec![Subscription {
                reads: FieldSet::of(&[Field::StandardLoggingPayloadRef]),
                ..Subscription::on(Hook::AttemptEnded)
            }],
            has_request_state: false,
        },
        kept: Arc::default(),
    })
}

fn stub_litellm(py: Python<'_>) {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../..");
    let locals = PyDict::new(py);
    locals.set_item("repo", repo.to_str().unwrap()).unwrap();
    if let Err(error) = py.run(&CString::new(STUB).unwrap(), Some(&locals), Some(&locals)) {
        error.display(py);
        panic!("the litellm stub script failed");
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap()
}

#[rstest]
fn a_payload_handed_by_reference_renders_as_safe_dumps_does_and_frees_on_release(
    keeper: Arc<Keeper>,
) {
    let kept = Arc::clone(&keeper.kept);
    Python::initialize();
    let locals = Python::attach(|py| {
        stub_litellm(py);
        let locals = PyDict::new(py);
        locals
            .set_item(
                "rust",
                Bound::new(py, RustCallback::new(keeper).unwrap()).unwrap(),
            )
            .unwrap();
        if let Err(error) = py.run(&CString::new(SCRIPT).unwrap(), Some(&locals), Some(&locals)) {
            error.display(py);
            panic!("the payload script failed");
        }
        locals.unbind()
    });
    let expected = |name: &str| {
        Python::attach(|py| {
            locals
                .bind(py)
                .get_item(name)
                .unwrap()
                .unwrap()
                .extract::<String>()
                .unwrap()
                .into_bytes()
        })
    };
    let refcount = || {
        Python::attach(|py| {
            py.import("sys")
                .unwrap()
                .call_method1("getrefcount", (locals.bind(py).get_item("copy").unwrap(),))
                .unwrap()
                .extract::<isize>()
                .unwrap()
        })
    };
    let payload = {
        let mut kept = kept.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(kept.at_hook, [Ok(expected("at_hook"))]);
        kept.payloads.pop().unwrap()
    };
    let handle = payload.handle();
    let runtime = runtime();

    let later = runtime.block_on(payload.render(RenderOptions::default()));
    let prompts_only = runtime.block_on(payload.render(RenderOptions {
        null_keys: vec!["messages".to_owned(), "response".to_owned()],
    }));
    let held = refcount();
    drop(payload);
    let released = runtime.block_on(payload::host().render(vec![handle], RenderOptions::default()));

    assert_eq!(later, Ok(expected("later")));
    assert_ne!(expected("later"), expected("at_hook"));
    assert_eq!(prompts_only, Ok(expected("prompts_only")));
    assert_eq!(released, Err("the payload was already released".to_owned()));
    assert_eq!(refcount(), held - 1);
}

#[rstest]
fn an_explicit_release_frees_the_handle_even_while_another_handle_is_held() {
    Python::initialize();
    let (first, second) = Python::attach(|py| {
        stub_litellm(py);
        (
            payload::stash(PyDict::new(py).into_any().unbind()),
            payload::stash(py.None()),
        )
    });
    let first_handle = first.handle();
    let runtime = runtime();

    first.release();

    assert_eq!(
        runtime.block_on(payload::host().render(vec![first_handle], RenderOptions::default())),
        Err("the payload was already released".to_owned())
    );
    assert_eq!(
        runtime.block_on(second.render(RenderOptions::default())),
        Ok(b"null".to_vec())
    );
}
