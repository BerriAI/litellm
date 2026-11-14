use litellm_db::Endpoint;
use litellm_secret_value::SecretValue;
use pyo3::{exceptions::PyValueError, prelude::*};
use sqlx::PgPool;

#[pyclass(frozen)]
pub(crate) struct NativeDatabase {
    writer_endpoint: Endpoint,
    writer: PgPool,
}

fn map_error(error: litellm_db::Error) -> PyErr {
    PyValueError::new_err(litellm_error::Report::new(&error).to_string())
}

#[pymethods]
impl NativeDatabase {
    #[staticmethod]
    fn connect(
        py: Python<'_>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] writer: Endpoint,
    ) -> PyResult<Bound<'_, PyAny>> {
        crate::execution::run_async_value(py, async move {
            let pool = writer.pool().map_err(map_error)?;
            Ok(NativeDatabase {
                writer_endpoint: writer,
                writer: pool,
            })
        })
    }

    fn set_writer_password(&self, password: String) {
        self.writer.set_connect_options(
            self.writer_endpoint
                .with_password(SecretValue::new(password))
                .connect_options(),
        );
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let writer = self.writer.clone();
        crate::execution::run_async(
            py,
            async move {
                writer.close().await;
                Ok(())
            },
            |error: PyErr| error,
        )
    }
}
