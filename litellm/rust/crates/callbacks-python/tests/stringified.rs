use std::{
    ffi::CString,
    path::Path,
    sync::{Arc, Mutex, PoisonError},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    BudgetReservation, Callback, Field, FieldSet, Hook, HookInput, HookOutput, LoggedKeyIdentity,
    Manifest, ReservationEntry, Scalar, Selection, Subscription, typed,
};
use litellm_callbacks_python::NativeCallback;
use pyo3::{prelude::*, types::PyDict};
use rstest::rstest;

type Seen = (
    Option<LoggedKeyIdentity>,
    Option<Vec<Option<Scalar>>>,
    Option<BudgetReservation>,
);

struct Recorder {
    manifest: Manifest,
    seen: Mutex<Option<Seen>>,
}

impl Callback for Recorder {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(&'a self, input: &'a HookInput) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            let seen = (
                input.fields.get::<typed::LoggedKeyIdentity>()?,
                input.fields.get::<typed::RequestTags>()?,
                input.fields.get::<typed::BudgetReservation>()?,
            );
            *self.seen.lock().unwrap_or_else(PoisonError::into_inner) = Some(seen);
            Ok(None)
        })
    }
}

const SCRIPT: &str = r#"
import asyncio
import sys
import types
from pathlib import Path

repo = Path(globals()["repo"])

for name, path in (("litellm", repo / "litellm"), ("litellm.integrations", repo / "litellm" / "integrations")):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.__path__ = [str(path)]
        sys.modules[name] = stub

from litellm.integrations.custom_logger_v2 import Field, Hook
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.derivations import budget_reservation, logged_key_identity, request_tags
from litellm.integrations.custom_logger_v2.native import native_logger
from litellm.integrations.custom_logger_v2.slots import HookArgs


class Marked:
    def __str__(self):
        return "litellm-raw-json:x"


request = {
    "standard_logging_object": {
        "metadata": {"user_api_key_team_id": ["t1"], "user_api_key_alias": Marked(), "user_api_key_org_id": set()},
        "request_tags": {"prod": 1, "dev": []},
    }
}
auth = types.SimpleNamespace(budget_reservation={"entries": ({"counter_key": "k"}, "x")})
view = types.SimpleNamespace(
    logged_key_identity=logged_key_identity(HookArgs(request=request)),
    request_tags=request_tags(HookArgs(request=request)),
    budget_reservation=budget_reservation(HookArgs(auth=auth)),
)
asyncio.run(
    native_logger(globals()["native"]).run_native(
        Hook.attempt_ended,
        frozenset({Field.logged_key_identity, Field.request_tags, Field.budget_reservation}),
        view,
        CallContext(hook=Hook.attempt_ended, call=None, state=None),
    )
)
"#;

fn stringified(text: &str, truthy: bool) -> Option<Scalar> {
    Some(Scalar::Stringified {
        text: text.to_owned(),
        truthy,
    })
}

#[rstest]
fn values_python_derives_reach_rust_as_their_str_text_truthiness_and_iteration() {
    let recorder = Arc::new(Recorder {
        manifest: Manifest {
            selection: Selection::named("stringified_recorder"),
            subscriptions: vec![Subscription {
                reads: FieldSet::of(&[
                    Field::LoggedKeyIdentity,
                    Field::RequestTags,
                    Field::BudgetReservation,
                ]),
                ..Subscription::on(Hook::AttemptEnded)
            }],
            has_request_state: false,
        },
        seen: Mutex::default(),
    });
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../..");
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item(
                "native",
                Bound::new(py, NativeCallback::new(Arc::clone(&recorder) as _).unwrap()).unwrap(),
            )
            .unwrap();
        locals.set_item("repo", repo.to_str().unwrap()).unwrap();
        if let Err(error) = py.run(&CString::new(SCRIPT).unwrap(), Some(&locals), Some(&locals)) {
            error.display(py);
            panic!("the stringified script failed");
        }
    });

    let (identity, tags, reservation) = recorder
        .seen
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
        .expect("the native ran");
    let identity = identity.unwrap();
    assert_eq!(identity.rejected, None);
    assert_eq!(identity.team_id, stringified("['t1']", true));
    assert_eq!(
        identity.key_alias,
        stringified("\u{f8ff}litellm-raw-json:x", true)
    );
    assert_eq!(identity.org_id, stringified("set()", false));
    assert_eq!(
        tags,
        Some(vec![
            Some(Scalar::Text("prod".to_owned())),
            Some(Scalar::Text("dev".to_owned()))
        ])
    );
    let reservation = reservation.unwrap();
    assert_eq!(reservation.entries_is_list, Some(false));
    assert_eq!(
        reservation.entries,
        Some(vec![
            Some(ReservationEntry {
                counter_key: Some("k".to_owned()),
                ..ReservationEntry::default()
            }),
            None
        ])
    );
}
