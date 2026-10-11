use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex, PoisonError},
    task::{Context, Poll, Waker},
};

use pyo3::{exceptions::PyException, prelude::*};

use crate::{Suspension, missing_state};

#[derive(Default)]
struct Exchange {
    suspension: Option<Suspension>,
    answer: Option<PyResult<Py<PyAny>>>,
}

/// The caller's Python task as the driver, hooks and host bindings see it. A body polled
/// through [`PythonTask::resume`] hands every awaitable, stream head and chunk back to that
/// task with [`PythonTask::suspend`], so Python work runs in the caller's task and context
/// and the body continues where it stopped once the caller resumes it.
#[derive(Clone)]
pub struct PythonTask {
    asynchronous: bool,
    exchange: Arc<Mutex<Exchange>>,
}

pub enum Resumed<T> {
    Suspended(Suspension),
    Complete(T),
}

impl PythonTask {
    pub fn new(asynchronous: bool) -> Self {
        Self {
            asynchronous,
            exchange: Arc::default(),
        }
    }

    pub fn asynchronous(&self) -> bool {
        self.asynchronous
    }

    fn exchange(&self) -> std::sync::MutexGuard<'_, Exchange> {
        self.exchange.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Hands `suspension` to the caller and resolves to what the caller resumes with.
    pub fn suspend(&self, suspension: Suspension) -> Suspend<'_> {
        Suspend {
            task: self,
            pending: Some(suspension),
        }
    }

    /// Settles what a Python call returned. On an asynchronous call the returned awaitable
    /// is awaited in the caller's task first. A cancellation (any `BaseException` that is
    /// not an `Exception`) is the outer error and ends the call; anything else is the inner
    /// result for the caller to decode, including an ordinary exception.
    pub async fn settle(&self, result: PyResult<Py<PyAny>>) -> PyResult<PyResult<Py<PyAny>>> {
        let result = match result {
            Ok(awaitable) if self.asynchronous => self.suspend(Suspension::Await(awaitable)).await,
            result => result,
        };
        Python::attach(|py| match result {
            Err(error) if !error.is_instance_of::<PyException>(py) => Err(error),
            result => Ok(result),
        })
    }

    /// Polls `future` once on the caller's task, `answer` being what the caller resumes the
    /// last suspension with. A future that pends without suspending broke the protocol.
    pub fn resume<F: Future + ?Sized>(
        &self,
        future: Pin<&mut F>,
        answer: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<Resumed<F::Output>> {
        {
            let mut exchange = self.exchange();
            exchange.answer = answer;
            exchange.suspension = None;
        }
        match future.poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(output) => Ok(Resumed::Complete(output)),
            Poll::Pending => self
                .exchange()
                .suspension
                .take()
                .map(Resumed::Suspended)
                .ok_or_else(missing_state),
        }
    }
}

pub struct Suspend<'a> {
    task: &'a PythonTask,
    pending: Option<Suspension>,
}

impl Future for Suspend<'_> {
    type Output = PyResult<Py<PyAny>>;

    fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        let mut exchange = self.task.exchange();
        if let Some(suspension) = self.pending.take() {
            exchange.suspension = Some(suspension);
            return Poll::Pending;
        }
        match exchange.answer.take() {
            Some(answer) => Poll::Ready(answer),
            None => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;

    use pyo3::exceptions::{PyValueError, asyncio::CancelledError};
    use rstest::rstest;

    use super::*;
    use crate::{InitializedPython, initialized_python};

    fn awaitable(py: Python<'_>, name: &str) -> Py<PyAny> {
        pyo3::types::PyString::new(py, name).into_any().unbind()
    }

    #[rstest]
    fn a_synchronous_task_settles_without_suspending(initialized_python: &InitializedPython) {
        initialized_python.attach(|py| {
            let task = PythonTask::new(false);
            let mut settled = pin!(task.settle(Ok(awaitable(py, "value"))));
            let Resumed::Complete(Ok(Ok(value))) = task.resume(settled.as_mut(), None).unwrap()
            else {
                panic!("expected the value without a suspension")
            };
            assert_eq!(value.extract::<String>(py).unwrap(), "value");
        });
    }

    #[rstest]
    fn an_asynchronous_task_suspends_on_the_awaitable_and_resumes_with_its_answer(
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let task = PythonTask::new(true);
            let mut settled = pin!(task.settle(Ok(awaitable(py, "pending"))));
            let Resumed::Suspended(Suspension::Await(suspended)) =
                task.resume(settled.as_mut(), None).unwrap()
            else {
                panic!("expected a suspension on the awaitable")
            };
            assert_eq!(suspended.extract::<String>(py).unwrap(), "pending");
            let Resumed::Complete(Ok(Ok(value))) = task
                .resume(settled.as_mut(), Some(Ok(awaitable(py, "answer"))))
                .unwrap()
            else {
                panic!("expected the answer")
            };
            assert_eq!(value.extract::<String>(py).unwrap(), "answer");
        });
    }

    #[rstest]
    #[case::cancellation(true)]
    #[case::ordinary_exception(false)]
    fn only_cancellations_end_a_settled_call(
        initialized_python: &InitializedPython,
        #[case] cancellation: bool,
    ) {
        initialized_python.attach(|py| {
            let task = PythonTask::new(true);
            let error = if cancellation {
                CancelledError::new_err("cancelled")
            } else {
                PyValueError::new_err("failed")
            };
            let raised = error.value(py).clone().unbind();
            let mut settled = pin!(task.settle(Ok(awaitable(py, "pending"))));
            task.resume(settled.as_mut(), None).unwrap();
            let Resumed::Complete(settled) =
                task.resume(settled.as_mut(), Some(Err(error))).unwrap()
            else {
                panic!("expected completion")
            };
            let delivered = match (cancellation, settled) {
                (true, Err(error)) | (false, Ok(Err(error))) => error,
                (_, other) => panic!("misrouted error: {other:?}"),
            };
            assert!(delivered.value(py).is(raised.bind(py)));
        });
    }

    #[rstest]
    fn a_body_that_pends_without_suspending_is_a_protocol_error(
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let task = PythonTask::new(true);
            let mut never = pin!(std::future::pending::<()>());
            assert!(task.resume(never.as_mut(), None).is_err());
            let mut immediate = pin!(async { 7 });
            assert!(matches!(
                task.resume(immediate.as_mut(), Some(Ok(py.None())))
                    .unwrap(),
                Resumed::Complete(7)
            ));
        });
    }
}
