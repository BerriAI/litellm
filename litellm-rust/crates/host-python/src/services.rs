use crate::{InvokeError, PythonOwned};
use litellm_host::protocol::Protocol;
use pyo3::prelude::*;

/// Capture Rust reply state only. Retained Python objects belong in the binding so
/// its PythonOwned implementation can expose those references to garbage collection.
pub type HostResume<E> = Box<
    dyn FnOnce(Python<'_>, PyResult<Py<PyAny>>) -> Result<HostStep<E>, InvokeError<E>>
        + Send
        + Sync,
>;

pub enum HostStep<E> {
    Ready,
    Await(Py<PyAny>, HostResume<E>),
}

impl<E> HostStep<E> {
    pub fn from_python(
        py: Python<'_>,
        asynchronous: bool,
        result: PyResult<Py<PyAny>>,
        resume: impl FnOnce(Python<'_>, PyResult<Py<PyAny>>) -> Result<Self, InvokeError<E>>
        + Send
        + Sync
        + 'static,
    ) -> Result<Self, InvokeError<E>> {
        match result {
            Ok(awaitable) if asynchronous => Ok(Self::Await(awaitable, Box::new(resume))),
            result => Self::resolve(py, result, resume),
        }
    }

    pub fn resolve(
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
        resume: impl FnOnce(Python<'_>, PyResult<Py<PyAny>>) -> Result<Self, InvokeError<E>>,
    ) -> Result<Self, InvokeError<E>> {
        match result {
            Err(error) if !error.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                Err(InvokeError::Python(error))
            }
            result => resume(py, result),
        }
    }
}

pub trait PythonHostCalls<P: Protocol>: PythonOwned {
    fn handle_host_call(
        &mut self,
        py: Python<'_>,
        call: P::HostCall,
    ) -> Result<(), InvokeError<P::Error>>;

    fn begin_host_call(
        &mut self,
        py: Python<'_>,
        call: P::HostCall,
    ) -> Result<HostStep<P::Error>, InvokeError<P::Error>> {
        self.handle_host_call(py, call).map(|()| HostStep::Ready)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InitializedPython, initialized_python};
    use litellm_host::protocol::reply;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    struct Dropped(Arc<AtomicBool>);

    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[rstest::rstest]
    #[case::immediate(false)]
    #[case::awaited(true)]
    fn host_results_settle_the_captured_reply(
        initialized_python: &InitializedPython,
        #[case] asynchronous: bool,
    ) {
        initialized_python.attach(|py| {
            let (reply, answer) = reply::<String>();
            let step = HostStep::<()>::from_python(
                py,
                asynchronous,
                Ok("value".into_pyobject(py).unwrap().into_any().unbind()),
                move |py, result| {
                    reply.send(result?.extract::<String>(py)?);
                    Ok(HostStep::Ready)
                },
            )
            .unwrap();
            let completed = match step {
                HostStep::Ready => HostStep::Ready,
                HostStep::Await(value, resume) => HostStep::resolve(py, Ok(value), resume).unwrap(),
            };
            assert!(matches!(completed, HostStep::Ready));
            assert_eq!(
                futures_util::FutureExt::now_or_never(answer)
                    .unwrap()
                    .unwrap(),
                "value"
            );
        });
    }

    #[rstest::rstest]
    #[case::ordinary_exception(false, false)]
    #[case::immediate_cancellation(false, true)]
    #[case::awaited_exception(true, false)]
    #[case::awaited_cancellation(true, true)]
    fn host_errors_preserve_cancellation_and_release_reply_state(
        initialized_python: &InitializedPython,
        #[case] asynchronous: bool,
        #[case] cancellation: bool,
    ) {
        initialized_python.attach(|py| {
            let error = if cancellation {
                pyo3::exceptions::asyncio::CancelledError::new_err("cancelled")
            } else {
                pyo3::exceptions::PyValueError::new_err("unavailable")
            };
            let exception = error.value(py).clone().unbind();
            let dropped = Arc::new(AtomicBool::new(false));
            let captured = Dropped(dropped.clone());
            let (reply, answer) = reply::<bool>();
            let resume = move |_: Python<'_>, result: PyResult<Py<PyAny>>| {
                let _retained = captured;
                reply.send(result.is_err());
                Ok(HostStep::<()>::Ready)
            };
            let outcome = if asynchronous {
                let step = HostStep::from_python(py, true, Ok(py.None()), resume).unwrap();
                let HostStep::Await(_, continuation) = step else {
                    panic!("expected suspension")
                };
                HostStep::resolve(py, Err(error), continuation)
            } else {
                HostStep::from_python(py, false, Err(error), resume)
            };
            assert!(dropped.load(Ordering::SeqCst));
            let delivered = futures_util::FutureExt::now_or_never(answer).unwrap();
            if cancellation {
                let Err(InvokeError::Python(error)) = outcome else {
                    panic!("expected cancellation")
                };
                assert!(error.value(py).is(exception.bind(py)));
                assert!(delivered.is_err());
            } else {
                assert!(matches!(outcome, Ok(HostStep::Ready)));
                assert!(delivered.unwrap());
            }
        });
    }

    #[rstest::rstest]
    fn dropping_a_suspended_host_step_abandons_its_reply(initialized_python: &InitializedPython) {
        initialized_python.attach(|py| {
            let (reply, answer) = reply::<()>();
            let step = HostStep::<()>::from_python(py, true, Ok(py.None()), move |_, result| {
                result?;
                reply.send(());
                Ok(HostStep::Ready)
            })
            .unwrap();
            drop(step);
            assert!(
                futures_util::FutureExt::now_or_never(answer)
                    .unwrap()
                    .is_err()
            );
        });
    }
}
