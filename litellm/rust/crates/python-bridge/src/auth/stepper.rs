use std::{
    future::Future,
    pin::pin,
    sync::{Mutex, MutexGuard, PoisonError},
    task::{Context, Poll, Waker},
};

use litellm_coroutine::{Co, Coroutine, CoroutineState};
use pyo3::{PyErr, PyResult, exceptions::PyRuntimeError};

struct State<Y, C> {
    coroutine: Coroutine<Y, C>,
    pending: Option<Y>,
}

/// Runs a core decision one host question at a time. Python calls `resume` to get the next
/// question or the outcome and answers each question before resuming. The core does no IO of its
/// own, so a single poll always reaches the next question or the end.
pub(crate) struct Stepper<Y, C>(Mutex<State<Y, C>>);

fn unexpected(answer: &str) -> PyErr {
    PyRuntimeError::new_err(format!("auth core is not waiting for {answer}"))
}

impl<Y, C> Stepper<Y, C> {
    pub(crate) fn new<F>(body: impl FnOnce(Co<Y>) -> F) -> Self
    where
        F: Future<Output = C> + Send + 'static,
    {
        Self(Mutex::new(State {
            coroutine: Coroutine::new(body),
            pending: None,
        }))
    }

    fn state(&self) -> MutexGuard<'_, State<Y, C>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn resume<S>(&self, describe: impl FnOnce(&Y) -> S) -> PyResult<Result<S, C>> {
        let mut state = self.state();
        if state.pending.is_some() {
            return Err(PyRuntimeError::new_err(
                "answer the pending auth core question first",
            ));
        }
        let polled = pin!(state.coroutine.resume()).poll(&mut Context::from_waker(Waker::noop()));
        match polled {
            Poll::Pending => Err(PyRuntimeError::new_err(
                "auth core waited on something other than its host",
            )),
            Poll::Ready(Err(error)) => Err(PyRuntimeError::new_err(error.to_string())),
            Poll::Ready(Ok(CoroutineState::Complete(outcome))) => Ok(Err(outcome)),
            Poll::Ready(Ok(CoroutineState::Yielded(ask))) => {
                let step = describe(&ask);
                state.pending = Some(ask);
                Ok(Ok(step))
            }
        }
    }

    /// Hands the pending question to `answer`, which returns it back when it is not the kind it
    /// answers, so a wrong answer leaves the question pending.
    pub(crate) fn answer(
        &self,
        kind: &str,
        answer: impl FnOnce(Y) -> Result<(), Y>,
    ) -> PyResult<()> {
        let mut state = self.state();
        let Some(pending) = state.pending.take() else {
            return Err(unexpected(kind));
        };
        answer(pending).map_err(|pending| {
            state.pending = Some(pending);
            unexpected(kind)
        })
    }
}
