use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};

use litellm_core::error::Error;
use pyo3::exceptions::PyLookupError;
use pyo3::types::PyDict;

use super::*;

mod synthetic {
    use std::future::{Future, pending};

    use super::*;

    static FUTURE_DROPPED: AtomicBool = AtomicBool::new(false);

    struct DropGuard;

    impl Drop for DropGuard {
        fn drop(&mut self) {
            FUTURE_DROPPED.store(true, Ordering::SeqCst);
        }
    }

    #[pyfunction]
    fn future_dropped() -> bool {
        FUTURE_DROPPED.load(Ordering::SeqCst)
    }

    struct EchoInputs {
        value: String,
    }

    #[pyfunction]
    fn echo(py: Python<'_>, value: String) -> PyResult<Py<PyAny>> {
        let future = prepare_echo(EchoInputs { value })?;
        litellm_python_interop::run_sync(py, future, map_error)
    }

    #[pyfunction]
    fn aecho(py: Python<'_>, value: String) -> PyResult<Bound<'_, PyAny>> {
        let future = prepare_echo(EchoInputs { value })?;
        litellm_python_interop::run_async(py, future, map_error)
    }

    pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        super::add_function(module, wrap_pyfunction!(future_dropped, module)?)?;
        super::add_function(module, wrap_pyfunction!(echo, module)?)?;
        super::add_function(module, wrap_pyfunction!(aecho, module)?)
    }

    #[cfg(feature = "trace-parity")]
    mod trace {
        use pyo3::prelude::*;

        use super::{EchoInputs, map_error, prepare_echo};

        #[pyfunction]
        fn echo(py: Python<'_>, value: String) -> PyResult<Py<PyAny>> {
            let future = prepare_echo(EchoInputs { value })?;
            litellm_python_interop::run_sync(py, crate::trace_parity::capture(future), map_error)
        }

        #[pyfunction]
        fn aecho(py: Python<'_>, value: String) -> PyResult<Bound<'_, PyAny>> {
            let future = prepare_echo(EchoInputs { value })?;
            litellm_python_interop::run_async(py, crate::trace_parity::capture(future), map_error)
        }

        pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
            super::super::add_function(module, wrap_pyfunction!(echo, module)?)?;
            super::super::add_function(module, wrap_pyfunction!(aecho, module)?)
        }
    }

    #[cfg(feature = "trace-parity")]
    pub(super) fn register_trace(module: &Bound<'_, PyModule>) -> PyResult<()> {
        trace::register(module)
    }

    fn prepare_echo(
        inputs: EchoInputs,
    ) -> PyResult<impl Future<Output = Result<String, Error>> + Send + 'static> {
        FUTURE_DROPPED.store(false, Ordering::SeqCst);
        let drop_guard = (inputs.value == "pending").then_some(DropGuard);
        Ok(execute_echo(inputs, drop_guard))
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn execute_echo(
        inputs: EchoInputs,
        drop_guard: Option<DropGuard>,
    ) -> Result<String, Error> {
        let _drop_guard = drop_guard;
        tokio::task::yield_now().await;
        match inputs.value.as_str() {
            "error" => Err(Error::InvalidRequest("synthetic error".to_string())),
            "map_panic" => Err(Error::InvalidRequest("panic in mapper".to_string())),
            "panic" => panic!("synthetic panic"),
            "pending" => {
                pending::<()>().await;
                unreachable!()
            }
            _ => Ok(inputs.value),
        }
    }

    fn map_error(error: Error) -> PyErr {
        if matches!(&error, Error::InvalidRequest(message) if message == "panic in mapper") {
            panic!("synthetic mapper panic")
        }
        PyLookupError::new_err(error.to_string())
    }
}

#[test]
fn generated_routes_execute_sync_and_async_contracts() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "synthetic").expect("module should be created");
        synthetic::register(&module).expect("routes should register");

        let sync_value: String = module
            .getattr("echo")
            .and_then(|function| function.call1(("sync",)))
            .and_then(|value| value.extract())
            .expect("sync route should return its value");
        assert_eq!(sync_value, "sync");

        let sync_error = module
            .getattr("echo")
            .and_then(|function| function.call1(("error",)))
            .expect_err("sync route should map its error");
        assert!(sync_error.is_instance_of::<PyLookupError>(py));
        assert_eq!(
            sync_error.to_string(),
            "LookupError: invalid request: synthetic error"
        );

        let locals = PyDict::new(py);
        locals
            .set_item("routes", &module)
            .expect("module should enter Python locals");
        let code = CString::new(
            r#"
import asyncio

async def exercise():
    assert await routes.aecho("async") == "async"

    try:
        await routes.aecho("error")
    except LookupError as error:
        assert str(error) == "invalid request: synthetic error"
    else:
        raise AssertionError("mapped error was not raised")

    try:
        await routes.aecho("panic")
    except BaseException as error:
        assert type(error).__name__ == "PanicException"
        assert str(error) == "synthetic panic"
    else:
        raise AssertionError("panic was not raised")

    try:
        await routes.aecho("map_panic")
    except BaseException as error:
        assert type(error).__name__ == "PanicException"
        assert str(error) == "synthetic mapper panic"
    else:
        raise AssertionError("mapper panic was not raised")

    task = asyncio.ensure_future(routes.aecho("pending"))
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancelled route completed")

    for _ in range(100):
        if routes.future_dropped():
            break
        await asyncio.sleep(0.001)
    assert routes.future_dropped()

asyncio.run(exercise())
"#,
        )
        .expect("Python source should not contain null bytes");
        py.run(&code, Some(&locals), Some(&locals))
            .expect("async route contract should hold");
    });
}

#[cfg(feature = "trace-parity")]
#[test]
fn diagnostic_route_returns_the_response_and_filtered_trace() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "synthetic").expect("module should be created");
        synthetic::register_trace(&module).expect("trace routes should register");
        let locals = PyDict::new(py);
        locals
            .set_item("routes", &module)
            .expect("module should enter Python locals");
        let code = CString::new(
            r#"
result = routes.echo("traced")
assert result["response"] == "traced", result
assert [event["function"] for event in result["trace"]] == ["execute_echo"], result
failure = routes.echo("error")
assert failure["error"] == "invalid request: synthetic error", failure
assert [event["function"] for event in failure["trace"]] == ["execute_echo"], failure
"#,
        )
        .expect("Python source should not contain null bytes");
        py.run(&code, Some(&locals), Some(&locals))
            .expect("diagnostic route should return its response and trace");
    });
}

#[test]
fn route_registration_rejects_duplicate_python_names() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "synthetic").expect("module should be created");
        synthetic::register(&module).expect("first registration should succeed");
        let error =
            synthetic::register(&module).expect_err("duplicate registration should be rejected");

        assert_eq!(
            error.to_string(),
            "RuntimeError: duplicate native route: future_dropped"
        );
    });
}
