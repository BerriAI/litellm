use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::time::Duration;

use futures_util::FutureExt;
use litellm_core::error::Error;
use litellm_python_interop::{Pythonized, panic_to_pyerr, release_gil, to_py};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::Serialize;
use tokio::runtime::{Handle, Runtime};
use tokio::time::{self, MissedTickBehavior};

pub(super) fn run_sync<T, F>(
    py: Python<'_>,
    future: F,
    map_error: fn(Error) -> PyErr,
) -> PyResult<Py<PyAny>>
where
    T: Serialize + Send + 'static,
    F: Future<Output = Result<T, Error>> + Send + 'static,
{
    run_sync_on(
        py,
        pyo3_async_runtimes::tokio::get_runtime(),
        future,
        map_error,
    )
}

fn run_sync_on<T, F>(
    py: Python<'_>,
    runtime: &Runtime,
    future: F,
    map_error: fn(Error) -> PyErr,
) -> PyResult<Py<PyAny>>
where
    T: Serialize + Send + 'static,
    F: Future<Output = Result<T, Error>> + Send + 'static,
{
    if Handle::try_current().is_ok() {
        return Err(PyRuntimeError::new_err(
            "synchronous native routes cannot run from a Tokio context; use the async route",
        ));
    }

    let result = release_gil(py, move || runtime.block_on(wait_for_sync_result(future)))?;
    let result = map_core_result(result, map_error)?;
    std::panic::catch_unwind(AssertUnwindSafe(|| to_py(py, &result))).map_err(panic_to_pyerr)?
}

pub(super) fn run_async<T, F>(
    py: Python<'_>,
    future: F,
    map_error: fn(Error) -> PyErr,
) -> PyResult<Bound<'_, PyAny>>
where
    T: Serialize + Send + 'static,
    F: Future<Output = Result<T, Error>> + Send + 'static,
{
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let result = catch_route_panic(future).await?;
        let result = map_core_result(result, map_error)?;
        Ok(Pythonized(result))
    })
}

fn map_core_result<T>(result: Result<T, Error>, map_error: fn(Error) -> PyErr) -> PyResult<T> {
    match result {
        Ok(value) => Ok(value),
        Err(error) => Err(
            std::panic::catch_unwind(AssertUnwindSafe(|| map_error(error)))
                .map_err(panic_to_pyerr)?,
        ),
    }
}

async fn catch_route_panic<T, F>(future: F) -> PyResult<Result<T, Error>>
where
    F: Future<Output = Result<T, Error>>,
{
    AssertUnwindSafe(future)
        .catch_unwind()
        .await
        .map_err(panic_to_pyerr)
}

async fn wait_for_sync_result<T, F>(future: F) -> PyResult<Result<T, Error>>
where
    F: Future<Output = Result<T, Error>>,
{
    let future = catch_route_panic(future);
    tokio::pin!(future);

    let signal_interval = Duration::from_millis(50);
    let mut signal_checks =
        time::interval_at(time::Instant::now() + signal_interval, signal_interval);
    signal_checks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            result = &mut future => return result,
            _ = signal_checks.tick() => Python::attach(|py| py.check_signals())?,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::future::poll_fn;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, mpsc};
    use std::task::Poll;
    use std::thread;
    use std::time::Instant;

    use pyo3::panic::PanicException;
    use pyo3::types::{PyDict, PyModule};
    use serde::Serializer;
    use tokio::runtime::Builder;

    use super::*;

    fn runtime_error(error: Error) -> PyErr {
        PyRuntimeError::new_err(error.to_string())
    }

    fn panicking_error_mapper(_error: Error) -> PyErr {
        panic!("error mapper panicked")
    }

    struct PanickingOutput;

    static ASYNC_PROBE_COMPLETED: AtomicUsize = AtomicUsize::new(0);

    impl Serialize for PanickingOutput {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            panic!("serializer panicked")
        }
    }

    #[pyfunction]
    fn async_serialization_panic(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        run_async(py, async { Ok(PanickingOutput) }, runtime_error)
    }

    #[pyfunction]
    fn async_runtime_probe(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        run_async(
            py,
            async {
                ASYNC_PROBE_COMPLETED.fetch_add(1, Ordering::SeqCst);
                Ok(true)
            },
            runtime_error,
        )
    }

    #[pyfunction]
    fn runtime_worker_count() -> usize {
        pyo3_async_runtimes::tokio::get_runtime()
            .metrics()
            .num_workers()
    }

    #[pyfunction]
    fn runtime_is_responsive(_py: Python<'_>, expected_completions: usize) -> bool {
        let completion_deadline = Instant::now() + Duration::from_secs(2);
        while ASYNC_PROBE_COMPLETED.load(Ordering::SeqCst) < expected_completions {
            if Instant::now() >= completion_deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(1));
        }

        let (heartbeat_tx, heartbeat_rx) = mpsc::sync_channel(1);
        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            let _ = heartbeat_tx.send(());
        });
        heartbeat_rx.recv_timeout(Duration::from_secs(2)).is_ok()
    }

    fn extract_bool(py: Python<'_>, result: PyResult<Py<PyAny>>) -> bool {
        result
            .expect("route should complete")
            .bind(py)
            .extract()
            .expect("result should convert")
    }

    #[test]
    fn sync_runner_polls_future_on_the_caller_thread() {
        Python::initialize();
        Python::attach(|py| {
            let caller_thread = std::thread::current().id();
            let result = run_sync(
                py,
                async move { Ok(std::thread::current().id() == caller_thread) },
                runtime_error,
            );

            assert!(extract_bool(py, result));
        });
    }

    #[test]
    fn sync_runner_releases_gil_while_waiting() {
        Python::initialize();
        Python::attach(|py| {
            let result = run_sync(
                py,
                async {
                    let gil_acquired = tokio::time::timeout(
                        Duration::from_secs(2),
                        tokio::task::spawn_blocking(|| Python::attach(|_| true)),
                    )
                    .await;
                    Ok(matches!(gil_acquired, Ok(Ok(true))))
                },
                runtime_error,
            );

            assert!(extract_bool(py, result));
        });
    }

    #[test]
    fn sync_runner_rejects_calls_from_a_tokio_context() {
        Python::initialize();
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime should build");

        let error = runtime.block_on(async {
            Python::attach(|py| {
                run_sync::<bool, _>(py, async { Ok(true) }, runtime_error)
                    .expect_err("sync route should reject a nested Tokio runtime")
            })
        });

        assert_eq!(
            error.to_string(),
            "RuntimeError: synchronous native routes cannot run from a Tokio context; use the async route"
        );
    }

    #[test]
    fn sync_runner_can_drive_a_current_thread_runtime() {
        Python::initialize();
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime should build");
        Python::attach(|py| {
            let result = run_sync_on(
                py,
                &runtime,
                async {
                    tokio::task::yield_now().await;
                    Ok(true)
                },
                runtime_error,
            );
            assert!(extract_bool(py, result));
        });
    }

    #[test]
    fn sync_runner_maps_a_panicked_future() {
        Python::initialize();
        Python::attach(|py| {
            let error = run_sync::<bool, _>(
                py,
                poll_fn(|_| -> Poll<Result<bool, Error>> { panic!("route future panicked") }),
                runtime_error,
            )
            .expect_err("panicked route should become a Python exception");

            assert!(error.is_instance_of::<PanicException>(py));
            assert_eq!(error.to_string(), "PanicException: route future panicked");
        });
    }

    #[test]
    fn sync_runner_maps_a_panicked_error_mapper() {
        Python::initialize();
        Python::attach(|py| {
            let error = run_sync::<bool, _>(
                py,
                async { Err(Error::InvalidRequest("invalid".to_string())) },
                panicking_error_mapper,
            )
            .expect_err("panicked mapper should become a Python exception");

            assert!(error.is_instance_of::<PanicException>(py));
            assert_eq!(error.to_string(), "PanicException: error mapper panicked");
        });
    }

    #[test]
    fn sync_runner_surfaces_serializer_panics() {
        Python::initialize();
        Python::attach(|py| {
            let error = run_sync(py, async { Ok(PanickingOutput) }, runtime_error)
                .expect_err("serializer panic should become a Python exception");

            assert!(error.is_instance_of::<PanicException>(py));
            assert_eq!(error.to_string(), "PanicException: serializer panicked");
        });
    }

    #[test]
    fn sync_runner_supports_concurrent_callers_on_the_shared_runtime() {
        Python::initialize();
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let callers: Vec<_> = (0..2)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    Python::attach(|py| {
                        extract_bool(
                            py,
                            run_sync(
                                py,
                                async move {
                                    Ok(tokio::time::timeout(Duration::from_secs(2), barrier.wait())
                                        .await
                                        .is_ok())
                                },
                                runtime_error,
                            ),
                        )
                    })
                })
            })
            .collect();
        let results: Vec<_> = callers
            .into_iter()
            .map(|caller| caller.join().expect("caller should not panic"))
            .collect();

        assert_eq!(results, vec![true, true]);
    }

    #[test]
    fn async_runner_surfaces_serializer_panics() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "runtime").expect("module should be created");
            module
                .add_function(
                    wrap_pyfunction!(async_serialization_panic, &module)
                        .expect("function should wrap"),
                )
                .expect("function should register");
            let locals = PyDict::new(py);
            locals
                .set_item("runtime", &module)
                .expect("module should enter Python locals");
            let code = CString::new(
                r#"
import asyncio

async def exercise():
    try:
        await runtime.async_serialization_panic()
    except BaseException as error:
        assert type(error).__name__ == "PanicException"
        assert str(error) == "serializer panicked"
    else:
        raise AssertionError("serializer panic was not raised")

asyncio.run(exercise())
"#,
            )
            .expect("Python source should not contain null bytes");
            py.run(&code, Some(&locals), Some(&locals))
                .expect("serializer panic should reach the Python awaiter");
        });
    }

    #[test]
    fn async_result_delivery_does_not_stall_tokio_workers() {
        Python::initialize();
        ASYNC_PROBE_COMPLETED.store(0, Ordering::SeqCst);
        Python::attach(|py| {
            let module = PyModule::new(py, "runtime").expect("module should be created");
            for function in [
                wrap_pyfunction!(async_runtime_probe, &module).expect("function should wrap"),
                wrap_pyfunction!(runtime_worker_count, &module).expect("function should wrap"),
                wrap_pyfunction!(runtime_is_responsive, &module).expect("function should wrap"),
            ] {
                module
                    .add_function(function)
                    .expect("function should register");
            }
            let locals = PyDict::new(py);
            locals
                .set_item("runtime", &module)
                .expect("module should enter Python locals");
            let code = CString::new(
                r#"
import asyncio

async def exercise():
    worker_count = runtime.runtime_worker_count()
    awaitables = [runtime.async_runtime_probe() for _ in range(worker_count)]
    assert runtime.runtime_is_responsive(worker_count)
    assert await asyncio.gather(*awaitables) == [True] * worker_count

asyncio.run(exercise())
"#,
            )
            .expect("Python source should not contain null bytes");
            py.run(&code, Some(&locals), Some(&locals))
                .expect("result delivery should leave Tokio workers responsive");
        });
    }
}
