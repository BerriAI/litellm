use std::{
    ffi::CString,
    sync::{Arc, Mutex, PoisonError},
};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Callback, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, RemainingUsage, RouterLimits,
    RouterReadError, RouterUsage, Selection, Subscription, UsageAmount,
};
use litellm_callbacks_python::{RustCallback, router::PyRouterRef};
use num_bigint::BigInt;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rstest::rstest;

struct Limited;

impl RouterUsage for Limited {
    fn remaining(
        &self,
        model_group: &str,
        limits: &RouterLimits,
    ) -> BoxFuture<'static, Result<RemainingUsage, RouterReadError>> {
        let headers = vec![
            (
                "x-ratelimit-remaining-tokens".to_owned(),
                UsageAmount::Int(BigInt::from(model_group.len())),
            ),
            (
                "x-ratelimit-limit-tokens".to_owned(),
                UsageAmount::Int(BigInt::from(10_u8).pow(40)),
            ),
            (
                "x-ratelimit-remaining-requests".to_owned(),
                UsageAmount::Float(2.5),
            ),
        ];
        let failure: Option<RouterReadError> = match model_group {
            "broken" => Some(Box::new(PyValueError::new_err("router boom"))),
            "text" => Some("no python error".into()),
            _ if limits.reset != Some(0) => Some("the limits did not arrive".into()),
            _ => None,
        };
        Box::pin(async move { failure.map_or(Ok(RemainingUsage { headers }), Err) })
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
                        .remaining(
                            "group",
                            &RouterLimits {
                                reset: Some(0),
                                ..RouterLimits::default()
                            },
                        )
                        .await
                        .map_err(|error| {
                            litellm_callbacks::CallbackError::Failed(error.to_string())
                        })?,
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
                headers: vec![
                    (
                        "x-ratelimit-remaining-tokens".to_owned(),
                        UsageAmount::Int(BigInt::from(5))
                    ),
                    (
                        "x-ratelimit-limit-tokens".to_owned(),
                        UsageAmount::Int(BigInt::from(10_u8).pow(40))
                    ),
                    (
                        "x-ratelimit-remaining-requests".to_owned(),
                        UsageAmount::Float(2.5)
                    ),
                ],
            }),
            None,
        ]
    );
    assert_eq!(refused, "router_ref must be a router reference, got object");
}

const AWAITED: &str = r#"
import asyncio


async def read(group):
    try:
        return await router.remaining(group, limits)
    except Exception as error:
        return type(error).__name__, str(error)


limits = {"reset": 0}
answered = asyncio.run(read("group"))
broken = asyncio.run(read("broken"))
text = asyncio.run(read("text"))
limits = {"reset": 0, "unknown": 1}
undecoded = asyncio.run(read("group"))


def read_sync(group, limits):
    try:
        return router.remaining_sync(group, limits)
    except Exception as error:
        return type(error).__name__, str(error)


blocked = read_sync("group", {"reset": 0})
blocked_broken = read_sync("broken", {"reset": 0})
blocked_text = read_sync("text", {"reset": 0})
"#;

#[rstest]
fn a_router_reference_awaited_from_python_answers_its_headers_or_raises_the_read_error() {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item(
                "router",
                Bound::new(py, PyRouterRef::new(Arc::new(Limited))).unwrap(),
            )
            .unwrap();
        if let Err(error) = py.run(
            &CString::new(AWAITED).unwrap(),
            Some(&locals),
            Some(&locals),
        ) {
            error.display(py);
            panic!("the awaited script failed");
        }
        let repr = |name: &str| {
            locals
                .get_item(name)
                .unwrap()
                .unwrap()
                .repr()
                .unwrap()
                .to_string()
        };
        assert_eq!(
            repr("answered"),
            "{'x-ratelimit-remaining-tokens': 5, 'x-ratelimit-limit-tokens': 10000000000000000000000000000000000000000, 'x-ratelimit-remaining-requests': 2.5}"
        );
        assert_eq!(repr("broken"), "('ValueError', 'router boom')");
        assert_eq!(repr("text"), "('RuntimeError', 'no python error')");
        assert!(
            repr("undecoded").contains("unknown field `unknown`"),
            "{}",
            repr("undecoded")
        );
        assert_eq!(repr("blocked"), repr("answered"));
        assert_eq!(repr("blocked_broken"), "('ValueError', 'router boom')");
        assert_eq!(repr("blocked_text"), "('RuntimeError', 'no python error')");
    });
}
