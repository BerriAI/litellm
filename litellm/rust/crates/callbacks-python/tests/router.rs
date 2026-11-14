use std::{
    ffi::CString,
    sync::{Arc, Mutex, PoisonError},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Callback, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, RemainingUsage, RouterUsage,
    Selection, Subscription, UsageAmount,
};
use litellm_callbacks_python::{RustCallback, router::PyRouterRef};
use pyo3::{prelude::*, types::PyDict};
use rstest::rstest;

struct Limited;

impl RouterUsage for Limited {
    fn remaining(&self, model_group: &str) -> BoxFuture<'static, Result<RemainingUsage, String>> {
        let headers = vec![(
            "x-ratelimit-remaining-tokens".to_owned(),
            UsageAmount::Int(i128::try_from(model_group.len()).unwrap_or_default()),
        )];
        Box::pin(async move { Ok(RemainingUsage { headers }) })
    }
}

struct Reader {
    manifest: Manifest,
    read: Arc<Mutex<Vec<Option<RemainingUsage>>>>,
}

impl Callback for Reader {
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
            let remaining = match input.fields.router()? {
                Some(router) => Some(
                    router
                        .remaining("group")
                        .await
                        .map_err(litellm_callbacks::CallbackError::Failed)?,
                ),
                None => None,
            };
            self.read
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(remaining);
            Ok(None)
        })
    }
}

const SCRIPT: &str = r#"
import asyncio


async def handle(value):
    return await rust.handle("attempt_ended", {"router_ref": value}, None, None)


asyncio.run(handle(router))
asyncio.run(handle(None))
try:
    asyncio.run(handle(object()))
except TypeError as error:
    refused = str(error)
"#;

#[rstest]
fn a_router_handed_by_reference_reaches_the_callback_and_anything_else_is_refused() {
    let read = Arc::default();
    let reader = Arc::new(Reader {
        manifest: Manifest {
            selection: Selection::named("router_reader"),
            subscriptions: vec![Subscription {
                reads: FieldSet::of(&[Field::RouterRef]),
                ..Subscription::on(Hook::AttemptEnded)
            }],
            has_request_state: false,
        },
        read: Arc::clone(&read),
    });
    Python::initialize();
    let refused = Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item(
                "rust",
                Bound::new(py, RustCallback::new(reader).unwrap()).unwrap(),
            )
            .unwrap();
        locals
            .set_item(
                "router",
                Bound::new(py, PyRouterRef::new(Arc::new(Limited))).unwrap(),
            )
            .unwrap();
        if let Err(error) = py.run(&CString::new(SCRIPT).unwrap(), Some(&locals), Some(&locals)) {
            error.display(py);
            panic!("the router script failed");
        }
        locals
            .get_item("refused")
            .unwrap()
            .unwrap()
            .extract::<String>()
            .unwrap()
    });

    assert_eq!(
        *read.lock().unwrap_or_else(PoisonError::into_inner),
        vec![
            Some(RemainingUsage {
                headers: vec![(
                    "x-ratelimit-remaining-tokens".to_owned(),
                    UsageAmount::Int(5)
                )],
            }),
            None,
        ]
    );
    assert_eq!(refused, "router_ref must be a router reference, got object");
}
