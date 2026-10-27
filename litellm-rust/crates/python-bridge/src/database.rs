use std::error::Error as StdError;

use litellm_db::Error;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use serde_json::Value;
use sqlx::{PgPool, postgres::PgPoolOptions};

fn map_error(error: Error) -> PyErr {
    let chain: Vec<String> =
        std::iter::successors(Some(&error as &dyn StdError), |error| error.source())
            .map(ToString::to_string)
            .collect();
    PyRuntimeError::new_err(chain.join(": "))
}

#[pyclass(frozen)]
pub struct NativeDatabase {
    pool: PgPool,
}

#[pymethods]
impl NativeDatabase {
    #[staticmethod]
    fn connect(py: Python<'_>, url: String, max_connections: u32) -> PyResult<Bound<'_, PyAny>> {
        crate::execution::run_async_value(py, async move {
            let pool = PgPoolOptions::new()
                .max_connections(max_connections)
                .connect(&url)
                .await
                .map_err(|error| map_error(error.into()))?;
            Ok(Self { pool })
        })
    }

    fn find_user_banner<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pool = self.pool.clone();
        crate::execution::run_async(
            py,
            async move { litellm_db::find_user_banner(&pool).await },
            map_error,
        )
    }

    fn save_user_banner<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] settings: Value,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pool = self.pool.clone();
        crate::execution::run_async(
            py,
            async move { litellm_db::save_user_banner(&pool, &settings).await },
            map_error,
        )
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pool = self.pool.clone();
        crate::execution::run_async(
            py,
            async move {
                pool.close().await;
                Ok::<(), Error>(())
            },
            map_error,
        )
    }
}
