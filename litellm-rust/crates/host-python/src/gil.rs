use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::prelude::*;

static GIL_RELEASES: AtomicU64 = AtomicU64::new(0);

/// Runs work detached from the interpreter and records the release.
///
/// `f` must not access Python state while the interpreter is detached.
pub fn release_gil<T, F>(py: Python<'_>, f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    GIL_RELEASES.fetch_add(1, Ordering::Relaxed);
    py.detach(f)
}

pub fn release_count() -> u64 {
    GIL_RELEASES.load(Ordering::Relaxed)
}

/// Runs Python work that may block, such as a secret manager read or a callback that does
/// I/O, on the runtime's blocking pool so the async workers stay free to poll other calls.
pub async fn attach_blocking<T, F>(f: F) -> T
where
    F: for<'py> FnOnce(Python<'py>) -> T + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(move || Python::attach(f)).await {
        Ok(value) => value,
        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
        Err(error) => panic!("the blocking pool dropped a Python call: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
    use rstest::{fixture, rstest};

    use super::attach_blocking;
    use crate::{InitializedPython, initialized_python, run_sync_value};

    #[fixture]
    fn namespace(#[from(initialized_python)] python: &InitializedPython) -> Py<PyDict> {
        python.attach(|py| {
            let namespace = PyDict::new(py);
            py.run(
                c"
import threading, time
finished = False
def work(seconds):
    global finished
    time.sleep(seconds)
    finished = True
def observe(expression):
    return eval(expression)
",
                Some(&namespace),
                None,
            )
            .unwrap();
            namespace.unbind()
        })
    }

    fn observe<T: for<'a, 'py> FromPyObject<'a, 'py, Error: std::fmt::Debug>>(
        namespace: &Py<PyDict>,
        py: Python<'_>,
        expression: &str,
    ) -> T {
        namespace
            .bind(py)
            .get_item("observe")
            .unwrap()
            .unwrap()
            .call1((expression,))
            .unwrap()
            .extract()
            .unwrap()
    }

    fn work(namespace: &Py<PyDict>, py: Python<'_>, seconds: f64) {
        namespace
            .bind(py)
            .get_item("work")
            .unwrap()
            .unwrap()
            .call1((seconds,))
            .unwrap();
    }

    fn shared(namespace: &Py<PyDict>) -> Py<PyDict> {
        Python::attach(|py| namespace.clone_ref(py))
    }

    #[rstest]
    #[tokio::test]
    async fn blocking_python_work_leaves_the_runtime_free_to_run_other_tasks(
        namespace: Py<PyDict>,
    ) {
        let (python_done, timer_done) = tokio::join!(
            attach_blocking(move |py| {
                work(&namespace, py, 0.3);
                Instant::now()
            }),
            async {
                tokio::time::sleep(Duration::from_millis(30)).await;
                Instant::now()
            }
        );
        assert!(
            timer_done < python_done,
            "the timer only finished after the Python call: the call ran inline on the worker"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn python_work_runs_off_the_thread_polling_the_future(namespace: Py<PyDict>) {
        let polling: u64 = Python::attach(|py| observe(&namespace, py, "threading.get_ident()"));

        let worker: u64 =
            attach_blocking(move |py| observe(&namespace, py, "threading.get_ident()")).await;

        assert_ne!(worker, polling);
    }

    #[rstest]
    #[tokio::test]
    async fn a_dropped_await_never_interrupts_the_python_call(namespace: Py<PyDict>) {
        let handle = shared(&namespace);
        let started = tokio::time::timeout(
            Duration::from_millis(10),
            attach_blocking(move |py| work(&handle, py, 0.1)),
        )
        .await;
        assert!(
            started.is_err(),
            "the await was dropped before the call returned"
        );

        tokio::time::sleep(Duration::from_millis(300)).await;
        let finished: bool = Python::attach(|py| observe(&namespace, py, "finished"));
        assert!(finished);
    }

    #[rstest]
    #[tokio::test]
    async fn a_panic_in_python_work_reaches_the_awaiting_task(
        #[from(initialized_python)] _python: &InitializedPython,
    ) {
        let joined =
            tokio::spawn(attach_blocking(|_| -> () { panic!("python work failed") })).await;
        let error = joined.expect_err("the panic propagates instead of being swallowed");
        assert!(error.is_panic());
    }

    #[rstest]
    fn a_sync_route_can_await_python_work_without_deadlocking_on_the_gil(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        let value = python
            .attach(|py| {
                run_sync_value(py, async {
                    tokio::time::timeout(
                        Duration::from_secs(5),
                        attach_blocking(|_| Python::version_str().len()),
                    )
                    .await
                    .map_err(|_| {
                        PyRuntimeError::new_err("the blocking call never re-acquired the GIL")
                    })
                })
            })
            .unwrap();
        assert!(value > 0);
    }
}
