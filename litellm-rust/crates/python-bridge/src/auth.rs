use std::num::NonZeroUsize;
use std::sync::Mutex;

use litellm_core::auth::{OperationFuture, SecretString, TokenCredential, TokenProvider};
use litellm_python_interop::callback_runtime::{
    AsyncContext, CallbackRuntime, CallbackZero, Direct, SyncContext,
};
use pyo3::prelude::*;

use crate::constants::AUTH_CALLBACK_CAPACITY;

#[pyclass(frozen)]
struct AuthCallbackRuntime(CallbackRuntime);

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let capacity = NonZeroUsize::new(AUTH_CALLBACK_CAPACITY)
        .expect("auth callback capacity is a positive constant");
    module.add(
        "__auth_callback_runtime__",
        AuthCallbackRuntime(CallbackRuntime::new(module, capacity)?),
    )
}

enum PythonTokenSession {
    Sync {
        callback: CallbackZero<String, Direct>,
        context: SyncContext,
    },
    Async {
        callback: CallbackZero<String, Direct>,
        context: AsyncContext,
    },
}

pub struct PythonTokenProvider {
    session: tokio::sync::Mutex<PythonTokenSession>,
    original_error: Mutex<Option<PyErr>>,
}

impl PythonTokenProvider {
    pub fn new(
        module: &Bound<'_, PyModule>,
        callable: Py<PyAny>,
        py: Python<'_>,
        asynchronous: bool,
    ) -> PyResult<Self> {
        let runtime = module
            .getattr("__auth_callback_runtime__")?
            .extract::<PyRef<'_, AuthCallbackRuntime>>()?
            .0
            .clone();
        let session = if asynchronous {
            PythonTokenSession::Async {
                callback: CallbackZero::new(callable.bind(py).clone())?,
                context: runtime.async_context(py)?,
            }
        } else {
            PythonTokenSession::Sync {
                callback: CallbackZero::new(callable.bind(py).clone())?,
                context: runtime.sync_context(py)?,
            }
        };
        Ok(Self {
            session: tokio::sync::Mutex::new(session),
            original_error: Mutex::new(None),
        })
    }

    pub fn take_original_error(&self) -> Option<PyErr> {
        self.original_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    async fn resolve(&self) -> Result<String, litellm_core::Error> {
        let result = match &mut *self.session.lock().await {
            PythonTokenSession::Sync { callback, context } => callback.call(context).await,
            PythonTokenSession::Async { callback, context } => callback.call(context).await,
        };
        match result {
            Ok(token) => Ok(token),
            Err(error) => {
                *self
                    .original_error
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
                Err(litellm_core::Error::Auth(
                    "caller-supplied token provider failed".into(),
                ))
            }
        }
    }
}

impl TokenProvider for PythonTokenProvider {
    fn token(&self) -> OperationFuture<'_, TokenCredential> {
        Box::pin(async move {
            self.resolve()
                .await
                .map(|token| TokenCredential::NoStore(SecretString::new(token)))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::sync::Arc;

    use pyo3::exceptions::PyRuntimeError;
    use pyo3::types::{PyDict, PyModule};

    use super::*;
    use crate::execution;

    #[pyclass]
    struct AuthHarness {
        module: Py<PyModule>,
    }

    #[pymethods]
    impl AuthHarness {
        fn resolve_sync(&self, py: Python<'_>, callable: Py<PyAny>) -> PyResult<Py<PyAny>> {
            let provider = Arc::new(PythonTokenProvider::new(
                self.module.bind(py),
                callable,
                py,
                false,
            )?);
            execution::run_sync(py, resolve_token(provider), |error| error)
        }

        fn resolve_async<'py>(
            &self,
            py: Python<'py>,
            callable: Py<PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let provider = Arc::new(PythonTokenProvider::new(
                self.module.bind(py),
                callable,
                py,
                true,
            )?);
            execution::run_async(py, resolve_token(provider), |error| error)
        }
    }

    async fn resolve_token(provider: Arc<PythonTokenProvider>) -> PyResult<String> {
        match provider.token().await {
            Ok(TokenCredential::NoStore(token)) => Ok(token.expose().to_owned()),
            Ok(TokenCredential::KnownExpiry(_)) => Err(PyRuntimeError::new_err(
                "Python token providers must return no-store credentials",
            )),
            Err(error) => Err(provider
                .take_original_error()
                .unwrap_or_else(|| PyRuntimeError::new_err(error.to_string()))),
        }
    }

    fn test_module(py: Python<'_>) -> Bound<'_, PyModule> {
        let module = PyModule::new(py, "auth_test").expect("module should be created");
        litellm_python_interop::callback_runtime::register(&module)
            .expect("callback runtime should register");
        register(&module).expect("auth runtime should register");
        module
            .add(
                "harness",
                AuthHarness {
                    module: module.clone().unbind(),
                },
            )
            .expect("harness should register");
        module
    }

    #[test]
    fn sync_provider_is_zero_argument_context_bound_and_preserves_errors() {
        Python::initialize();
        Python::attach(|py| {
            let module = test_module(py);
            let locals = PyDict::new(py);
            locals
                .set_item("auth", &module)
                .expect("module should enter Python locals");
            let code = CString::new(
                r#"
import contextvars
import gc
import threading
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor

scope = contextvars.ContextVar("scope")
scope.set("caller-context")
caller_thread = threading.get_ident()
calls = 0

def token(*args):
    global calls
    calls += 1
    assert args == ()
    assert scope.get() == "caller-context"
    assert threading.get_ident() == caller_thread
    return "sync-secret"

assert auth.harness.resolve_sync(token) == "sync-secret"
assert calls == 1

class RetainedToken:
    def __call__(self):
        return "retained"

retained = RetainedToken()
reference = weakref.ref(retained)
assert auth.harness.resolve_sync(retained) == "retained"
del retained
gc.collect()
assert reference() is None

def resolve_one(index):
    expected = f"worker-{index}"
    scope.set(expected)
    caller = threading.get_ident()
    invocation_count = 0

    def concurrent_token(*args):
        nonlocal invocation_count
        invocation_count += 1
        assert args == ()
        assert scope.get() == expected
        assert threading.get_ident() == caller
        return expected

    assert auth.harness.resolve_sync(concurrent_token) == expected
    assert invocation_count == 1
    return expected

with ThreadPoolExecutor(max_workers=8) as executor:
    assert list(executor.map(resolve_one, range(16))) == [
        f"worker-{index}" for index in range(16)
    ]

original = LookupError("original-token-error")
def fail():
    raise original

try:
    auth.harness.resolve_sync(fail)
except BaseException as error:
    assert error is original
    assert "fail" in "".join(traceback.format_tb(error.__traceback__))
else:
    raise AssertionError("original exception was not raised")
"#,
            )
            .expect("Python source should not contain null bytes");
            py.run(&code, Some(&locals), Some(&locals))
                .expect("sync provider behavior should hold");
        });
    }

    #[test]
    fn async_providers_preserve_context_lifetime_concurrency_and_operation_count() {
        Python::initialize();
        Python::attach(|py| {
            let module = test_module(py);
            let locals = PyDict::new(py);
            locals
                .set_item("auth", &module)
                .expect("module should enter Python locals");
            let code = CString::new(
                r#"
import asyncio
import contextvars
import gc
import weakref

scope = contextvars.ContextVar("scope")

class Token:
    def __init__(self, expected):
        self.expected = expected
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        assert args == ()
        assert scope.get() == self.expected
        asyncio.get_running_loop()
        return f"token-{self.expected}"

async def resolve_one(index):
    expected = f"context-{index}"
    scope.set(expected)
    provider = Token(expected)
    value = await auth.harness.resolve_async(provider)
    assert provider.calls == 1
    return value

async def exercise():
    values = await asyncio.gather(*(resolve_one(index) for index in range(32)))
    assert values == [f"token-context-{index}" for index in range(32)]

    scope.set("repeat")
    repeated = Token("repeat")
    assert await auth.harness.resolve_async(repeated) == "token-repeat"
    assert await auth.harness.resolve_async(repeated) == "token-repeat"
    assert repeated.calls == 2

    scope.set("lifetime")
    retained = Token("lifetime")
    reference = weakref.ref(retained)
    pending = auth.harness.resolve_async(retained)
    del retained
    gc.collect()
    assert reference() is not None
    assert await pending == "token-lifetime"
    for _ in range(3):
        gc.collect()
        await asyncio.sleep(0)
    assert reference() is None

    original = RuntimeError("async-original-token-error")
    def fail():
        raise original
    try:
        await auth.harness.resolve_async(fail)
    except BaseException as error:
        assert error is original
    else:
        raise AssertionError("original async exception was not raised")

    try:
        await auth.harness.resolve_async(lambda: 42)
    except TypeError as error:
        assert str(error) == "hook result does not match its typed contract"
    else:
        raise AssertionError("invalid token result was accepted")

asyncio.run(exercise())
"#,
            )
            .expect("Python source should not contain null bytes");
            py.run(&code, Some(&locals), Some(&locals))
                .expect("async provider behavior should hold");
        });
    }
}
