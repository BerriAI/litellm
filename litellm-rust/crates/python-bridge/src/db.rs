use litellm_db::{Endpoint, Pools};
use litellm_secret_value::SecretValue;
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass(frozen)]
pub(crate) struct NativeDatabase {
    writer_endpoint: Endpoint,
    reader_endpoint: Option<Endpoint>,
    pools: Pools,
}

fn map_error(error: litellm_db::Error) -> PyErr {
    PyValueError::new_err(litellm_error::Report::new(&error).to_string())
}

#[pymethods]
impl NativeDatabase {
    #[staticmethod]
    #[pyo3(signature = (writer, reader=None))]
    fn connect(
        py: Python<'_>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] writer: Endpoint,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] reader: Option<Endpoint>,
    ) -> PyResult<Bound<'_, PyAny>> {
        crate::execution::run_async_value(py, async move {
            let pools = Pools::new(&writer, reader.as_ref()).map_err(map_error)?;
            Ok(NativeDatabase {
                writer_endpoint: writer,
                reader_endpoint: reader,
                pools,
            })
        })
    }

    fn set_writer_password(&self, password: String) {
        self.pools.writer().set_connect_options(
            self.writer_endpoint
                .with_password(SecretValue::new(password))
                .connect_options(),
        );
    }

    fn set_reader_password(&self, password: String) -> PyResult<()> {
        let (Some(endpoint), Some(pool)) = (&self.reader_endpoint, self.pools.reader_pool()) else {
            return Err(PyValueError::new_err("no read replica is configured"));
        };
        pool.set_connect_options(
            endpoint
                .with_password(SecretValue::new(password))
                .connect_options(),
        );
        Ok(())
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pools = self.pools.clone();
        crate::execution::run_async(
            py,
            async move {
                pools.close().await;
                Ok(())
            },
            |error: PyErr| error,
        )
    }
}
