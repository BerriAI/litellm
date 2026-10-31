use std::str::FromStr;

use litellm_db::{EntitySpend, add_key_spend};
use litellm_host_python::run_async_value;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use sqlx::postgres::{PgConnectOptions, PgPool, PgPoolOptions};

#[pyclass(frozen)]
pub(crate) struct NativeDatabase {
    pool: PgPool,
}

fn connect_options() -> PyResult<PgConnectOptions> {
    let url = std::env::var("DATABASE_URL")
        .map_err(|_| PyRuntimeError::new_err("DATABASE_URL is not set"))?;
    let base = url.split('?').next().unwrap_or(&url);
    PgConnectOptions::from_str(base)
        .map(|options| options.application_name("litellm-native"))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pymethods]
impl NativeDatabase {
    #[staticmethod]
    fn connect(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let options = connect_options()?;
        run_async_value(py, async move {
            let pool = PgPoolOptions::new()
                .max_connections(5)
                .connect_lazy_with(options);
            Ok(NativeDatabase { pool })
        })
    }

    fn add_key_spend<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<String>,
        costs: Vec<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pool = self.pool.clone();
        run_async_value(py, async move {
            add_key_spend(&pool, &EntitySpend { ids, costs })
                .await
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pool = self.pool.clone();
        run_async_value(py, async move {
            pool.close().await;
            Ok(())
        })
    }
}
