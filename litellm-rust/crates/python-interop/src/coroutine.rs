use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyTypeError};
use pyo3::prelude::*;

pub enum CoroutineStep {
    Return(Py<PyAny>),
    Await(Py<PyAny>),
}

pub trait CoroutineBody: Send {
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<CoroutineStep>;
}

#[pyclass]
pub struct PythonCoroutine {
    body: Option<Box<dyn CoroutineBody + Sync>>,
    iterator: Option<Py<PyAny>>,
    started: bool,
}

impl PythonCoroutine {
    pub fn new(body: impl CoroutineBody + Sync + 'static) -> Self {
        Self {
            body: Some(Box::new(body)),
            iterator: None,
            started: false,
        }
    }

    fn advance(
        &mut self,
        py: Python<'_>,
        mut result: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<Py<PyAny>> {
        if !self.started {
            if matches!(&result, Some(Ok(value)) if !value.is_none(py)) {
                return Err(PyTypeError::new_err(
                    "cannot send non-None value to a just-started coroutine",
                ));
            }
            self.started = true;
            if matches!(&result, Some(Ok(value)) if value.is_none(py)) {
                result = None;
            } else if let Some(Err(error)) = result {
                self.body = None;
                return Err(error);
            }
        }
        loop {
            if let Some(iterator) = &self.iterator {
                let yielded = match result.take() {
                    Some(Ok(value)) => iterator.call_method1(py, "send", (value,)),
                    Some(Err(error)) => iterator.call_method1(py, "throw", (error.value(py),)),
                    None => iterator.call_method0(py, "__next__"),
                };
                match yielded {
                    Ok(value) => return Ok(value),
                    Err(error) => {
                        self.iterator = None;
                        result = Some(if error.is_instance_of::<PyStopIteration>(py) {
                            error.value(py).getattr("value").map(Bound::unbind)
                        } else {
                            Err(error)
                        });
                    }
                }
            }
            let step = self
                .body
                .as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("cannot reuse already awaited coroutine"))?
                .resume(result.take());
            match step {
                Ok(CoroutineStep::Await(awaitable)) => {
                    match awaitable.call_method0(py, "__await__") {
                        Ok(iterator) => self.iterator = Some(iterator),
                        Err(error) => result = Some(Err(error)),
                    }
                }
                Ok(CoroutineStep::Return(value)) => {
                    self.body = None;
                    return Err(PyStopIteration::new_err((value,)));
                }
                Err(error) => {
                    self.body = None;
                    return Err(error);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    struct AwaitBody(Option<Py<PyAny>>);

    impl CoroutineBody for AwaitBody {
        fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<CoroutineStep> {
            if let Some(awaitable) = self.0.take() {
                assert!(result.is_none());
                return Ok(CoroutineStep::Await(awaitable));
            }
            result.expect("awaited outcome").map(CoroutineStep::Return)
        }
    }

    #[pyfunction]
    fn await_in_rust(awaitable: Py<PyAny>) -> PythonCoroutine {
        PythonCoroutine::new(AwaitBody(Some(awaitable)))
    }

    #[test]
    fn forwards_awaitables_in_the_caller_task_and_preserves_exceptions() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item(
                    "await_in_rust",
                    wrap_pyfunction!(await_in_rust, py).unwrap(),
                )
                .unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    r#"
import asyncio
from contextvars import ContextVar

async def exercise():
    caller = asyncio.current_task()
    context = ContextVar('test', default='before')
    result = object()
    async def operation():
        assert asyncio.current_task() is caller
        context.set('after')
        await asyncio.sleep(0)
        return result
    wrapped = await_in_rust(operation())
    try:
        wrapped.send(1)
    except TypeError:
        pass
    else:
        raise AssertionError('accepted initial value')
    assert await wrapped is result
    assert context.get() == 'after'
    try:
        await wrapped
    except RuntimeError:
        pass
    else:
        raise AssertionError('accepted reuse')

    failure = ValueError('identity')
    async def failing():
        await asyncio.sleep(0)
        raise failure
    try:
        await await_in_rust(failing())
    except ValueError as error:
        assert error is failure
    else:
        raise AssertionError('lost failure')

    entered = asyncio.Event()
    cleaned = []
    async def pending():
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)
    async def call():
        await await_in_rust(pending())
    task = asyncio.create_task(call())
    await entered.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError('lost cancellation')
    assert cleaned == [True]

asyncio.run(exercise())
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
        });
    }
}

#[pymethods]
impl PythonCoroutine {
    fn __await__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.advance(py, None)
    }

    fn send(&mut self, py: Python<'_>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
        self.advance(py, Some(Ok(value)))
    }

    fn throw(&mut self, py: Python<'_>, error: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.advance(py, Some(Err(PyErr::from_value(error))))
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let result = self
            .iterator
            .take()
            .map(|iterator| iterator.call_method0(py, "close"));
        self.body = None;
        result.transpose().map(|_| ())
    }
}
