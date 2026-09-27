use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use crate::fork_gate::{ForkGate, Refused, RuntimeAlreadyStarted};
use crate::{Pythonized, panic_to_pyerr, release_gil};
use futures_util::FutureExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::Serialize;
use tokio::runtime::{Handle, Runtime};
use tokio::time::{self, MissedTickBehavior};

pyo3::create_exception!(
    _native,
    ForkedAfterNativeRuntimeStarted,
    PyRuntimeError,
    "This process was forked after the native runtime started. Runtime threads do not survive fork(), so native routes cannot run here."
);

pyo3::create_exception!(
    _native,
    ProcessReservedForForking,
    PyRuntimeError,
    "This process was reserved for forking workers, so native routes cannot run here."
);

static FORK_GATE: ForkGate = ForkGate::new();

/// Whether this process has entered process-bound native execution.
pub fn runtime_started() -> bool {
    FORK_GATE.started(std::process::id())
}

/// Declares that this process exists to fork workers, so it must never start the runtime.
/// Fails if it already has. Workers are unaffected: the reservation is keyed by pid.
pub fn reserve_process_for_forking() -> Result<(), RuntimeAlreadyStarted> {
    FORK_GATE.reserve(std::process::id())
}

/// Claims process-bound native state before runtime startup or tokenizer execution.
pub fn enter_native() -> PyResult<()> {
    FORK_GATE
        .enter(std::process::id())
        .map_err(|refused| match refused {
            Refused::ReservedForForking => ProcessReservedForForking::new_err(
                "this process is reserved for forking workers and cannot run native routes; \
                 move the call into a worker, after the fork",
            ),
            Refused::ForkedAfterStart => ForkedAfterNativeRuntimeStarted::new_err(
                "this process was forked after the native runtime started, and runtime threads \
                 do not survive fork(); start workers with spawn or forkserver, or fork before \
                 the first native call",
            ),
        })
}

#[expect(clippy::disallowed_methods, reason = "this is the gated door")]
pub(crate) fn runtime() -> PyResult<&'static Runtime> {
    enter_native()?;
    Ok(pyo3_async_runtimes::tokio::get_runtime())
}

#[expect(clippy::disallowed_methods, reason = "this is the gated door")]
fn future_into_py<F, T>(py: Python<'_>, future: F) -> PyResult<Bound<'_, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'py> IntoPyObject<'py> + Send + 'static,
{
    enter_native()?;
    pyo3_async_runtimes::tokio::future_into_py(py, future)
}

pub fn run_sync<T, E, F>(
    py: Python<'_>,
    future: F,
    map_error: fn(E) -> PyErr,
) -> PyResult<Py<PyAny>>
where
    T: Serialize + Send + 'static,
    E: Send + 'static,
    F: Future<Output = Result<T, E>> + Send + 'static,
{
    run_sync_on(py, runtime()?, future, map_error)
}

pub fn run_sync_value<T, F>(py: Python<'_>, future: F) -> PyResult<T>
where
    T: Send + 'static,
    F: Future<Output = PyResult<T>> + Send + 'static,
{
    run_sync_value_on(py, runtime()?, future)
}

fn run_sync_value_on<T, F>(py: Python<'_>, runtime: &Runtime, future: F) -> PyResult<T>
where
    T: Send + 'static,
    F: Future<Output = PyResult<T>> + Send + 'static,
{
    if Handle::try_current().is_ok() {
        return Err(PyRuntimeError::new_err(
            "synchronous native routes cannot run from a Tokio context; use the async route",
        ));
    }
    release_gil(py, move || runtime.block_on(wait_for_sync_result(future)))?
}

fn run_sync_on<T, E, F>(
    py: Python<'_>,
    runtime: &Runtime,
    future: F,
    map_error: fn(E) -> PyErr,
) -> PyResult<Py<PyAny>>
where
    T: Serialize + Send + 'static,
    E: Send + 'static,
    F: Future<Output = Result<T, E>> + Send + 'static,
{
    if Handle::try_current().is_ok() {
        return Err(PyRuntimeError::new_err(
            "synchronous native routes cannot run from a Tokio context; use the async route",
        ));
    }

    let result = release_gil(py, move || runtime.block_on(wait_for_sync_result(future)))?;
    let result = map_core_result(result, map_error)?;
    Pythonized(result).into_pyobject(py).map(Bound::unbind)
}

pub fn run_async<T, E, F>(
    py: Python<'_>,
    future: F,
    map_error: fn(E) -> PyErr,
) -> PyResult<Bound<'_, PyAny>>
where
    T: Serialize + Send + 'static,
    E: Send + 'static,
    F: Future<Output = Result<T, E>> + Send + 'static,
{
    future_into_py(py, async move {
        let result = catch_future_panic(future).await?;
        let result = map_core_result(result, map_error)?;
        Ok(Pythonized(result))
    })
}

pub fn run_async_value<T, F>(py: Python<'_>, future: F) -> PyResult<Bound<'_, PyAny>>
where
    T: for<'py> IntoPyObject<'py> + Send + 'static,
    F: Future<Output = PyResult<T>> + Send + 'static,
{
    future_into_py(py, async move { catch_future_panic(future).await? })
}

pub fn poll_async_value<T, F>(py: Python<'_>, future: Pin<&mut F>) -> PyResult<Poll<T>>
where
    T: Send,
    F: Future<Output = PyResult<T>> + Send,
{
    let runtime = runtime()?;
    let result = release_gil(py, || {
        let _runtime = runtime.enter();
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            future.poll(&mut Context::from_waker(Waker::noop()))
        }))
        .map_err(panic_to_pyerr)
    })?;
    match result {
        Poll::Ready(result) => result.map(Poll::Ready),
        Poll::Pending => Ok(Poll::Pending),
    }
}

fn map_core_result<T, E>(result: Result<T, E>, map_error: fn(E) -> PyErr) -> PyResult<T> {
    match result {
        Ok(value) => Ok(value),
        Err(error) => Err(
            std::panic::catch_unwind(AssertUnwindSafe(|| map_error(error)))
                .map_err(panic_to_pyerr)?,
        ),
    }
}

async fn catch_future_panic<T, E, F>(future: F) -> PyResult<Result<T, E>>
where
    F: Future<Output = Result<T, E>>,
{
    AssertUnwindSafe(future)
        .catch_unwind()
        .await
        .map_err(panic_to_pyerr)
}

async fn wait_for_sync_result<T, E, F>(future: F) -> PyResult<Result<T, E>>
where
    F: Future<Output = Result<T, E>>,
{
    let future = catch_future_panic(future);
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
    use std::future::{pending, poll_fn};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, mpsc};
    use std::task::Poll;
    use std::thread;
    use std::time::Instant;

    use pyo3::exceptions::PyLookupError;
    use pyo3::panic::PanicException;
    use pyo3::types::{PyDict, PyModule};
    use rstest::rstest;
    use serde::Serializer;
    use tokio::runtime::Builder;

    use super::*;
    use crate::{InitializedPython, initialized_python};

    #[derive(Debug)]
    struct Error(String);

    impl std::fmt::Display for Error {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(&self.0)
        }
    }

    fn runtime_error(error: Error) -> PyErr {
        PyRuntimeError::new_err(error.to_string())
    }

    fn panicking_error_mapper(_error: Error) -> PyErr {
        panic!("error mapper panicked")
    }

    static ECHO_FUTURE_DROPPED: AtomicBool = AtomicBool::new(false);

    struct EchoDropGuard;

    impl Drop for EchoDropGuard {
        fn drop(&mut self) {
            ECHO_FUTURE_DROPPED.store(true, Ordering::SeqCst);
        }
    }

    fn echo_error(error: Error) -> PyErr {
        if error.0 == "panic in mapper" {
            panic!("error mapper panicked")
        }
        PyLookupError::new_err(error.0)
    }

    #[pyfunction]
    fn async_echo(py: Python<'_>, value: String) -> PyResult<Bound<'_, PyAny>> {
        ECHO_FUTURE_DROPPED.store(false, Ordering::SeqCst);
        let drop_guard = (value == "pending").then_some(EchoDropGuard);
        run_async(
            py,
            async move {
                let _drop_guard = drop_guard;
                tokio::task::yield_now().await;
                match value.as_str() {
                    "error" => Err(Error("mapped error".into())),
                    "map_panic" => Err(Error("panic in mapper".into())),
                    "panic" => panic!("route future panicked"),
                    "pending" => {
                        pending::<()>().await;
                        unreachable!()
                    }
                    _ => Ok(value),
                }
            },
            echo_error,
        )
    }

    #[pyfunction]
    fn echo_future_dropped() -> bool {
        ECHO_FUTURE_DROPPED.load(Ordering::SeqCst)
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
    fn runtime_worker_count() -> PyResult<usize> {
        Ok(runtime()?.metrics().num_workers())
    }

    #[pyfunction]
    fn runtime_is_responsive(_py: Python<'_>, expected_completions: usize) -> PyResult<bool> {
        let completion_deadline = Instant::now() + Duration::from_secs(2);
        while ASYNC_PROBE_COMPLETED.load(Ordering::SeqCst) < expected_completions {
            if Instant::now() >= completion_deadline {
                return Ok(false);
            }
            thread::sleep(Duration::from_millis(1));
        }

        let (heartbeat_tx, heartbeat_rx) = mpsc::sync_channel(1);
        runtime()?.spawn(async move {
            let _ = heartbeat_tx.send(());
        });
        Ok(heartbeat_rx.recv_timeout(Duration::from_secs(2)).is_ok())
    }

    fn extract_bool(py: Python<'_>, result: PyResult<Py<PyAny>>) -> bool {
        result
            .expect("route should complete")
            .bind(py)
            .extract()
            .expect("result should convert")
    }

    #[rstest]
    fn reaching_the_runtime_marks_the_process_as_started(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        python.attach(|py| {
            run_sync_value(py, async { Ok(()) }).unwrap();
            assert!(runtime_started());
        });
    }

    #[rstest]
    fn inline_poll_releases_gil_and_enters_runtime(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        python.attach(|py| {
            let (sender, receiver) = mpsc::sync_channel(1);
            let worker = thread::spawn(move || Python::attach(|_| sender.send(()).unwrap()));
            let mut future = Box::pin(async move {
                receiver.recv_timeout(Duration::from_secs(2)).unwrap();
                Ok(Handle::try_current().is_ok())
            });
            assert_eq!(
                poll_async_value(py, future.as_mut()).unwrap(),
                Poll::Ready(true)
            );
            worker.join().unwrap();
        });
    }

    #[rstest]
    fn inline_poll_contains_panics_and_preserves_python_errors(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        python.attach(|py| {
            let mut panicking = Box::pin(poll_fn(|_| -> Poll<PyResult<()>> {
                panic!("inline native panic")
            }));
            let error = poll_async_value(py, panicking.as_mut()).unwrap_err();
            assert!(error.is_instance_of::<PanicException>(py));
            let original = PyRuntimeError::new_err("inline failure");
            let identity = original.value(py).clone().unbind();
            let mut failing = Box::pin(async move { Err::<(), _>(original) });
            let error = poll_async_value(py, failing.as_mut()).unwrap_err();
            assert!(error.value(py).is(identity.bind(py)));
        });
    }

    #[pyfunction]
    fn pending_after_inline_poll(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let starts = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&starts);
        let mut future = Box::pin(async move {
            starts.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(5)).await;
            Ok(starts.load(Ordering::SeqCst))
        });
        assert!(poll_async_value(py, future.as_mut())?.is_pending());
        assert_eq!(observed.load(Ordering::SeqCst), 1);
        run_async_value(py, future)
    }

    #[rstest]
    fn inline_pending_future_resumes_on_tokio_without_restarting(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        python.attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item(
                    "pending",
                    wrap_pyfunction!(pending_after_inline_poll, py).unwrap(),
                )
                .unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    "import asyncio\nasync def exercise():\n    assert await asyncio.wait_for(pending(), 2) == 1\nasyncio.run(exercise())"
                ),
                Some(&locals),
                Some(&locals),
            ).unwrap();
        });
    }

    #[rstest]
    fn sync_runner_polls_future_on_the_caller_thread(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        python.attach(|py| {
            let caller_thread = std::thread::current().id();
            let result = run_sync(
                py,
                async move { Ok(std::thread::current().id() == caller_thread) },
                runtime_error,
            );

            assert!(extract_bool(py, result));
        });
    }

    #[rstest]
    fn sync_runner_releases_gil_while_waiting(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        python.attach(|py| {
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

    #[rstest]
    fn sync_runner_rejects_calls_from_a_tokio_context(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime should build");

        let error = runtime.block_on(async {
            python.attach(|py| {
                run_sync::<bool, Error, _>(py, async { Ok(true) }, runtime_error)
                    .expect_err("sync route should reject a nested Tokio runtime")
            })
        });

        assert_eq!(
            error.to_string(),
            "RuntimeError: synchronous native routes cannot run from a Tokio context; use the async route"
        );
    }

    #[rstest]
    fn sync_runner_can_drive_a_current_thread_runtime(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime should build");
        python.attach(|py| {
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

    #[rstest]
    fn sync_runner_maps_a_panicked_future(#[from(initialized_python)] python: &InitializedPython) {
        python.attach(|py| {
            let error = run_sync::<bool, Error, _>(
                py,
                poll_fn(|_| -> Poll<Result<bool, Error>> { panic!("route future panicked") }),
                runtime_error,
            )
            .expect_err("panicked route should become a Python exception");

            assert!(error.is_instance_of::<PanicException>(py));
            assert_eq!(error.to_string(), "PanicException: route future panicked");
        });
    }

    #[rstest]
    fn sync_runner_maps_a_panicked_error_mapper(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        python.attach(|py| {
            let error = run_sync::<bool, Error, _>(
                py,
                async { Err(Error("invalid".to_string())) },
                panicking_error_mapper,
            )
            .expect_err("panicked mapper should become a Python exception");

            assert!(error.is_instance_of::<PanicException>(py));
            assert_eq!(error.to_string(), "PanicException: error mapper panicked");
        });
    }

    #[rstest]
    fn sync_runner_surfaces_serializer_panics(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        python.attach(|py| {
            let error = run_sync(py, async { Ok(PanickingOutput) }, runtime_error)
                .expect_err("serializer panic should become a Python exception");

            assert!(error.is_instance_of::<PanicException>(py));
            assert_eq!(error.to_string(), "PanicException: serializer panicked");
        });
    }

    #[rstest]
    fn sync_runner_supports_concurrent_callers_on_the_shared_runtime(
        #[from(initialized_python)] _python: &InitializedPython,
    ) {
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

    #[rstest]
    fn async_runner_surfaces_serializer_panics(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        python.attach(|py| {
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

    #[rstest]
    fn async_result_delivery_does_not_stall_tokio_workers(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        ASYNC_PROBE_COMPLETED.store(0, Ordering::SeqCst);
        python.attach(|py| {
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

    #[rstest]
    fn async_runner_delivers_values_and_errors_and_drops_cancelled_futures(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        python.attach(|py| {
            let module = PyModule::new(py, "runtime").expect("module should be created");
            for function in [
                wrap_pyfunction!(async_echo, &module).expect("function should wrap"),
                wrap_pyfunction!(echo_future_dropped, &module).expect("function should wrap"),
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
    assert await runtime.async_echo("value") == "value"

    try:
        await runtime.async_echo("error")
    except LookupError as error:
        assert str(error) == "mapped error"
    else:
        raise AssertionError("mapped error was not raised")

    try:
        await runtime.async_echo("panic")
    except BaseException as error:
        assert type(error).__name__ == "PanicException"
        assert str(error) == "route future panicked"
    else:
        raise AssertionError("panic was not raised")

    try:
        await runtime.async_echo("map_panic")
    except BaseException as error:
        assert type(error).__name__ == "PanicException"
        assert str(error) == "error mapper panicked"
    else:
        raise AssertionError("mapper panic was not raised")

    task = asyncio.ensure_future(runtime.async_echo("pending"))
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancelled route completed")

    for _ in range(100):
        if runtime.echo_future_dropped():
            break
        await asyncio.sleep(0.001)
    assert runtime.echo_future_dropped()

asyncio.run(exercise())
"#,
            )
            .expect("Python source should not contain null bytes");
            py.run(&code, Some(&locals), Some(&locals))
                .expect("async route contract should hold");
        });
    }
}
