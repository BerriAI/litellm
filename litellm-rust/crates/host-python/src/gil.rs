use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use pyo3::{exceptions::PyRuntimeError, prelude::*};

/// The caller's `contextvars` context, captured at the Python entry point so blocking Python
/// work started from Rust sees the same request-local values as the Python caller.
#[derive(Clone)]
pub struct PythonContext(Arc<Py<PyAny>>);

impl PythonContext {
    pub fn capture(py: Python<'_>) -> PyResult<Self> {
        Ok(Self(Arc::new(
            py.import("contextvars")?
                .call_method0("copy_context")?
                .unbind(),
        )))
    }

    /// Runs `f` inside a fresh copy of the captured context. The copy is what lets two blocking
    /// calls run concurrently: a `contextvars.Context` cannot be entered twice at once.
    pub fn enter<T, F>(&self, py: Python<'_>, f: F) -> PyResult<T>
    where
        F: FnOnce(Python<'_>) -> T + Send + Sync + 'static,
        T: Send + 'static,
    {
        let copy = self.0.bind(py).call_method0("copy")?;
        let body = Arc::new(Mutex::new(Some(f)));
        let value = Arc::new(Mutex::new(None::<T>));
        let callback = {
            let body = Arc::clone(&body);
            let value = Arc::clone(&value);
            pyo3::types::PyCFunction::new_closure(py, None, None, move |_, _| {
                Python::attach(|py| -> PyResult<()> {
                    let body = body
                        .lock()
                        .expect("context body slot poisoned")
                        .take()
                        .expect("the context body ran more than once");
                    *value.lock().expect("context value slot poisoned") = Some(body(py));
                    Ok(())
                })
            })?
        };
        copy.call_method1("run", (callback,))?;
        value
            .lock()
            .expect("context value slot poisoned")
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("the context body produced no value"))
    }
}

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
/// The work runs inside a copy of `context` so request-local `contextvars` survive the hop.
pub async fn attach_blocking<T, F>(context: PythonContext, f: F) -> PyResult<T>
where
    F: for<'py> FnOnce(Python<'py>) -> T + Send + Sync + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(move || Python::attach(|py| context.enter(py, f))).await {
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

    use super::{PythonContext, attach_blocking};
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

    fn context() -> PythonContext {
        Python::attach(|py| PythonContext::capture(py).unwrap())
    }

    #[fixture]
    fn request_context() -> (PythonContext, Py<PyAny>) {
        Python::initialize();
        Python::attach(|py| {
            let namespace = PyDict::new(py);
            py.run(
                c"import contextvars\nrequest_var = contextvars.ContextVar('request_var', default='unset')",
                Some(&namespace),
                None,
            )
            .unwrap();
            let var = namespace.get_item("request_var").unwrap().unwrap();
            var.call_method1("set", ("request-value",)).unwrap();
            (PythonContext::capture(py).unwrap(), var.unbind())
        })
    }

    #[rstest]
    #[tokio::test]
    async fn blocking_python_work_leaves_the_runtime_free_to_run_other_tasks(
        namespace: Py<PyDict>,
    ) {
        let (python_done, timer_done) = tokio::join!(
            attach_blocking(context(), move |py| {
                work(&namespace, py, 0.3);
                Instant::now()
            }),
            async {
                tokio::time::sleep(Duration::from_millis(30)).await;
                Instant::now()
            }
        );
        assert!(
            timer_done < python_done.unwrap(),
            "the timer only finished after the Python call: the call ran inline on the worker"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn python_work_runs_off_the_thread_polling_the_future(namespace: Py<PyDict>) {
        let polling: u64 = Python::attach(|py| observe(&namespace, py, "threading.get_ident()"));

        let worker: u64 = attach_blocking(context(), move |py| {
            observe(&namespace, py, "threading.get_ident()")
        })
        .await
        .unwrap();

        assert_ne!(worker, polling);
    }

    #[rstest]
    #[tokio::test]
    async fn a_dropped_await_never_interrupts_the_python_call(namespace: Py<PyDict>) {
        let handle = shared(&namespace);
        let started = tokio::time::timeout(
            Duration::from_millis(10),
            attach_blocking(context(), move |py| work(&handle, py, 0.1)),
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
        let joined = tokio::spawn(attach_blocking(context(), |_| -> () {
            panic!("python work failed")
        }))
        .await;
        let error = joined.expect_err("the panic propagates instead of being swallowed");
        assert!(error.is_panic());
    }

    #[rstest]
    fn a_sync_route_can_await_python_work_without_deadlocking_on_the_gil(
        #[from(initialized_python)] python: &InitializedPython,
    ) {
        let value = python
            .attach(|py| {
                let context = PythonContext::capture(py).unwrap();
                run_sync_value(py, async move {
                    tokio::time::timeout(
                        Duration::from_secs(5),
                        attach_blocking(context, |_| Python::version_str().len()),
                    )
                    .await
                    .map_err(|_| {
                        PyRuntimeError::new_err("the blocking call never re-acquired the GIL")
                    })
                })
            })
            .unwrap()
            .unwrap();
        assert!(value > 0);
    }

    #[rstest]
    #[tokio::test]
    async fn blocking_work_sees_the_callers_contextvars(
        request_context: (PythonContext, Py<PyAny>),
    ) {
        let (context, var) = request_context;

        let seen: String = attach_blocking(context, move |py| {
            var.bind(py)
                .call_method0("get")
                .unwrap()
                .extract::<String>()
                .unwrap()
        })
        .await
        .unwrap();

        assert_eq!(seen, "request-value");
    }

    #[rstest]
    #[tokio::test]
    async fn concurrent_blocking_calls_each_enter_a_context_copy(
        request_context: (PythonContext, Py<PyAny>),
    ) {
        let (context, var) = request_context;
        let first = Python::attach(|py| var.clone_ref(py));
        let second = var;

        let (first_seen, second_seen) = tokio::join!(
            attach_blocking(context.clone(), move |py| {
                first
                    .bind(py)
                    .call_method0("get")
                    .unwrap()
                    .extract::<String>()
                    .unwrap()
            }),
            attach_blocking(context, move |py| {
                second
                    .bind(py)
                    .call_method0("get")
                    .unwrap()
                    .extract::<String>()
                    .unwrap()
            }),
        );

        assert_eq!(first_seen.unwrap(), "request-value");
        assert_eq!(second_seen.unwrap(), "request-value");
    }

    #[rstest]
    #[tokio::test]
    async fn writes_inside_blocking_work_do_not_leak_back_to_the_caller(
        request_context: (PythonContext, Py<PyAny>),
    ) {
        let (context, var) = request_context;
        let leaked = Python::attach(|py| var.clone_ref(py));

        attach_blocking(context, move |py| {
            var.bind(py).call_method1("set", ("worker-value",)).unwrap();
        })
        .await
        .unwrap();

        let caller_value: String = Python::attach(|py| {
            leaked
                .bind(py)
                .call_method0("get")
                .unwrap()
                .extract()
                .unwrap()
        });
        assert_ne!(caller_value, "worker-value");
    }
}
