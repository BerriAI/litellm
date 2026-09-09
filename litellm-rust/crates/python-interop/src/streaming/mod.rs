mod reader;

pub use reader::{ByteStreamReader, PythonByteStream, PythonStreamCompletion};

use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub fn to_py_bytes<'py>(py: Python<'py>, bytes: &[u8]) -> Bound<'py, PyBytes> {
    PyBytes::new(py, bytes)
}

#[pyclass]
pub struct AsyncByteStream {
    reader: ByteStreamReader,
}

impl AsyncByteStream {
    pub fn new(stream: PythonByteStream, completion: PythonStreamCompletion) -> Self {
        Self {
            reader: ByteStreamReader::new(stream, completion),
        }
    }
}

impl Drop for AsyncByteStream {
    fn drop(&mut self) {
        self.reader.close();
    }
}

#[pymethods]
impl AsyncByteStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let reader = self.reader.clone();
        Ok(crate::run_async_py(py, async move {
            match reader.next_chunk().await? {
                Some(bytes) => Python::attach(|py| Ok(to_py_bytes(py, &bytes).unbind())),
                None => Err(PyStopAsyncIteration::new_err(())),
            }
        })?
        .unbind())
    }

    fn aclose(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.reader.close();
        let reader = self.reader.clone();
        Ok(crate::run_async_py(py, async move { reader.aclose().await })?.unbind())
    }
}
