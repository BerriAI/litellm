use std::time::Duration;

use litellm_db::{AuthRead, AuthSpend, DatabasePools, DatabaseSettings};
use litellm_host_python::{Pythonized, ready_future, run_async_value};
use litellm_secret_value::SecretValue;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyNone,
};

#[pyclass(frozen)]
pub(crate) struct NativeDatabase {
    pools: Option<DatabasePools>,
}

fn db_error(error: litellm_db::Error) -> PyErr {
    match error {
        litellm_db::Error::InvalidSetting(_) | litellm_db::Error::InvalidAuthArguments => {
            PyValueError::new_err(litellm_error::Report::new(&error).to_string())
        }
        _ => PyRuntimeError::new_err(litellm_error::Report::new(&error).to_string()),
    }
}

fn duration(seconds: f64, name: &'static str) -> PyResult<Duration> {
    Duration::try_from_secs_f64(seconds)
        .map_err(|_| db_error(litellm_db::Error::InvalidSetting(name)))
}

fn settings(value: &Bound<'_, PyAny>) -> PyResult<DatabaseSettings> {
    let ssl_mode: String = value.getattr("ssl_mode")?.extract()?;
    let idle_timeout: Option<f64> = value.getattr("idle_timeout")?.extract()?;
    Ok(DatabaseSettings {
        host: value.getattr("host")?.extract()?,
        port: value.getattr("port")?.extract()?,
        username: value.getattr("username")?.extract()?,
        password: value
            .getattr("password")?
            .extract::<Option<String>>()?
            .map(SecretValue::from),
        database: value.getattr("database")?.extract()?,
        schema: value.getattr("schema")?.extract()?,
        ssl_mode: ssl_mode
            .parse()
            .map_err(|_| db_error(litellm_db::Error::InvalidSetting("ssl_mode")))?,
        ssl_root_cert: value.getattr("ssl_root_cert")?.extract()?,
        max_connections: value.getattr("max_connections")?.extract()?,
        acquire_timeout: duration(
            value.getattr("acquire_timeout")?.extract()?,
            "acquire_timeout",
        )?,
        connect_timeout: duration(
            value.getattr("connect_timeout")?.extract()?,
            "connect_timeout",
        )?,
        idle_timeout: idle_timeout
            .map(|seconds| duration(seconds, "idle_timeout"))
            .transpose()?,
        statement_cache_capacity: value.getattr("statement_cache_capacity")?.extract()?,
    })
}

#[pymethods]
impl NativeDatabase {
    #[staticmethod]
    #[pyo3(signature = (writer_settings=None, reader_settings=None))]
    fn connect<'py>(
        py: Python<'py>,
        writer_settings: Option<&Bound<'py, PyAny>>,
        reader_settings: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let writer = writer_settings.map(settings).transpose()?;
        let reader = reader_settings.map(settings).transpose()?;
        let Some(writer) = writer else {
            if reader.is_some() {
                return Err(db_error(litellm_db::Error::InvalidSetting(
                    "writer_settings",
                )));
            }
            return ready_future(py, Bound::new(py, Self { pools: None })?.as_any());
        };
        run_async_value(py, async move {
            let pools = DatabasePools::connect(writer, reader)
                .await
                .map_err(db_error)?;
            Ok(Self { pools: Some(pools) })
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(pools) = self.pools.clone() else {
            return ready_future(py, PyNone::get(py).as_any());
        };
        run_async_value(py, async move {
            pools.close().await;
            Ok(())
        })
    }

    #[getter]
    fn reader_available(&self) -> bool {
        self.pools
            .as_ref()
            .is_some_and(DatabasePools::reader_available)
    }

    #[pyo3(signature = (writer_password=None, reader_password=None))]
    fn set_passwords(
        &self,
        writer_password: Option<String>,
        reader_password: Option<String>,
    ) -> PyResult<()> {
        self.pools
            .as_ref()
            .ok_or_else(|| db_error(litellm_db::Error::NotConfigured))?
            .set_passwords(
                writer_password.map(SecretValue::from),
                reader_password.map(SecretValue::from),
            )
            .map_err(db_error)
    }

    #[pyo3(signature = (kind, ids, writer=false))]
    fn auth_read<'py>(
        &self,
        py: Python<'py>,
        kind: &str,
        ids: Vec<String>,
        writer: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kind: AuthRead = kind
            .parse()
            .map_err(|_| db_error(litellm_db::Error::InvalidAuthArguments))?;
        let pools = self
            .pools
            .clone()
            .ok_or_else(|| db_error(litellm_db::Error::NotConfigured))?;
        run_async_value(py, async move {
            let mut connection = pools.acquire(writer).await.map_err(db_error)?;
            let rows = litellm_db::auth_read(&mut *connection, kind, &ids)
                .await
                .map_err(db_error)?;
            Ok(Pythonized(
                rows.into_iter().map(|row| row.data).collect::<Vec<_>>(),
            ))
        })
    }

    fn auth_spend<'py>(
        &self,
        py: Python<'py>,
        kind: &str,
        id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kind: AuthSpend = kind
            .parse()
            .map_err(|_| db_error(litellm_db::Error::InvalidAuthArguments))?;
        let pools = self
            .pools
            .clone()
            .ok_or_else(|| db_error(litellm_db::Error::NotConfigured))?;
        run_async_value(py, async move {
            let mut connection = pools.acquire(false).await.map_err(db_error)?;
            litellm_db::auth_spend(&mut *connection, kind, &id)
                .await
                .map_err(db_error)
        })
    }
}
