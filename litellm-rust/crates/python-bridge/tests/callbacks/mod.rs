use std::ffi::{CStr, CString};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use litellm_python_interop::callback_runtime::CallbackRuntime;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

const PYTHON_TEST_SOURCE: &CStr = pyo3::ffi::c_str!(include_str!("test_callbacks.py"));

macro_rules! provider_catalog {
    ($consumer:path, $($options:tt)*) => {
        $consumer! {
            $($options)*
            {
                pre_request: PreRequest(crate::callback_tests::domain::Request)
                    -> crate::callback_tests::domain::Request = awaitable;
                pre_api_call: PreApiCall(crate::callback_tests::domain::BeforeSend)
                    -> () = direct;
                post_response: PostResponse(crate::callback_tests::domain::Response)
                    -> crate::callback_tests::domain::Response = awaitable;
                failure: Failure(crate::callback_tests::domain::ProviderFailure)
                    -> () = direct;
            }
        }
    };
}

macro_rules! direct_catalog {
    ($consumer:path, $($options:tt)*) => {
        $consumer! {
            $($options)*
            {
                transform: Transform(crate::callback_tests::domain::Request)
                    -> crate::callback_tests::domain::Request = direct;
            }
        }
    };
}

mod domain {
    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Request {
        pub text: String,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Response {
        pub text: String,
    }

    #[derive(Serialize)]
    pub struct BeforeSend {
        pub body: Request,
    }

    #[derive(Serialize)]
    pub struct ProviderFailure {
        pub message: String,
    }

    provider_catalog!(litellm_core::define_hooks, pub trait ProviderHooks;);
    direct_catalog!(litellm_core::define_hooks, pub trait DirectHooks;);

    pub enum CallError<E> {
        Hook(E),
        Provider {
            error: ProviderFailure,
            observer_error: Option<E>,
        },
    }

    struct PreparedCall(BeforeSend);
    struct ReadyCall(Request);

    impl PreparedCall {
        async fn finish_hooks<H: ProviderHooks>(
            self,
            hooks: &mut H,
        ) -> Result<ReadyCall, H::Error> {
            hooks.pre_api_call(&self.0).await?;
            Ok(ReadyCall(self.0.body))
        }
    }

    impl ReadyCall {
        fn send(self, calls: &AtomicUsize, fail: bool) -> Result<Response, ProviderFailure> {
            calls.fetch_add(1, Ordering::SeqCst);
            if fail {
                return Err(ProviderFailure {
                    message: "provider failed".into(),
                });
            }
            Ok(Response {
                text: format!("processed:{}", self.0.text),
            })
        }
    }

    pub async fn execute<H: ProviderHooks>(
        hooks: &mut H,
        calls: &AtomicUsize,
        fail: bool,
    ) -> Result<Response, CallError<H::Error>> {
        let request = hooks
            .pre_request(&Request {
                text: "input".into(),
            })
            .await
            .map_err(CallError::Hook)?;
        let ready = PreparedCall(BeforeSend { body: request })
            .finish_hooks(hooks)
            .await
            .map_err(CallError::Hook)?;
        let response = match ready.send(calls, fail) {
            Ok(response) => response,
            Err(error) => {
                let observer_error = hooks.failure(&error).await.err();
                return Err(CallError::Provider {
                    error,
                    observer_error,
                });
            }
        };
        hooks
            .post_response(&response)
            .await
            .map_err(CallError::Hook)
    }
}

use domain::{DirectHooks, ProviderHooks};

provider_catalog!(crate::bind_python_hooks,
    struct PythonProviderSession;
    trait domain::ProviderHooks;
);
direct_catalog!(crate::bind_python_hooks,
    struct PythonDirectSession;
    trait domain::DirectHooks;
);

fn map_error(error: domain::CallError<PyErr>) -> PyErr {
    match error {
        domain::CallError::Hook(error) => error,
        domain::CallError::Provider {
            error,
            observer_error,
        } => Python::attach(|py| {
            let exception = PyRuntimeError::new_err(error.message);
            if let Some(observer_error) = observer_error {
                exception
                    .value(py)
                    .setattr("observer_error", observer_error.value(py))
                    .expect("exception should retain its observer error");
            }
            exception
        }),
    }
}

#[pyclass(frozen)]
struct Harness {
    runtime: CallbackRuntime,
    calls: Arc<AtomicUsize>,
}

#[pymethods]
impl Harness {
    #[pyo3(signature = (adapter, fail=false))]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        adapter: &Bound<'py, PyAny>,
        fail: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut session = PythonProviderSession::new(adapter, self.runtime.async_context(py)?)?;
        let calls = Arc::clone(&self.calls);
        crate::execution::run_async(
            py,
            async move { domain::execute(&mut session, &calls, fail).await },
            map_error,
        )
    }

    fn sync(&self, py: Python<'_>, adapter: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let mut session = PythonDirectSession::new(adapter, self.runtime.sync_context(py)?)?;
        crate::execution::run_sync(
            py,
            async move {
                session
                    .transform(&domain::Request {
                        text: "input".into(),
                    })
                    .await
            },
            std::convert::identity,
        )
    }

    fn interrupt<'py>(
        &self,
        py: Python<'py>,
        adapter: &Bound<'py, PyAny>,
        stop: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut session = PythonProviderSession::new(adapter, self.runtime.async_context(py)?)?;
        let stop = pyo3_async_runtimes::into_future_with_locals(
            &pyo3_async_runtimes::tokio::get_current_locals(py)?,
            stop,
        )?;
        crate::execution::run_async(
            py,
            async move {
                let request = domain::Request {
                    text: "input".into(),
                };
                tokio::select! {
                    result = session.pre_request(&request) => { result?; },
                    result = stop => { result?; },
                }
                session.pre_request(&request).await
            },
            std::convert::identity,
        )
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

fn run_python_test(name: &str, capacity: usize) {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "callback_test").expect("test module should load");
        litellm_python_interop::callback_runtime::register(&module).expect("shim should register");
        let runtime = CallbackRuntime::new(&module, NonZeroUsize::new(capacity).unwrap())
            .expect("runtime should initialize");
        let harness = Harness {
            runtime,
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let tests = PyModule::from_code(
            py,
            PYTHON_TEST_SOURCE,
            c"callback_tests.py",
            &CString::new(format!("callback_tests_{name}")).unwrap(),
        )
        .expect("test definitions should load");
        tests
            .call_method1("run", (name, harness))
            .expect("callback contract should hold");
    });
}

macro_rules! python_tests {
    ($($name:ident: $capacity:literal,)*) => {
        $(
            #[test]
            fn $name() { run_python_test(stringify!($name), $capacity); }
        )*
    };
}

python_tests! {
    transforms_and_context: 64,
    callback_errors: 64,
    retained_callbacks: 64,
    registration_and_return_contracts: 64,
    provider_failure: 64,
    cancellation_and_admission: 1,
    interrupted_session: 1,
    synchronous_callbacks: 1,
}
