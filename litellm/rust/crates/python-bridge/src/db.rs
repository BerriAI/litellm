use std::{sync::Arc, time::UNIX_EPOCH};

use litellm_db::{Call, Endpoint, Observer, Outcome, Pools, QueryEvent};
use litellm_secret_value::SecretValue;
use pyo3::{exceptions::PyValueError, prelude::*};

const HEALTH_CHECK: Call = Call {
    call_type: "health_check",
    table: None,
    operation: None,
};

struct PythonEvents(Py<PyAny>);

impl Observer for PythonEvents {
    fn record(&self, event: QueryEvent) {
        let started_at = event
            .started_at
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let failure = match event.outcome {
            Outcome::Succeeded => None,
            Outcome::Failed(message) => Some(message),
        };
        Python::attach(|py| {
            let events = self.0.bind(py);
            if let Err(error) = events.call_method1(
                "record",
                (
                    event.call.call_type,
                    event.call.table,
                    event.call.operation,
                    started_at,
                    event.duration.as_secs_f64(),
                    failure,
                ),
            ) {
                error.write_unraisable(py, Some(events));
            }
        });
    }
}

#[pyclass(frozen)]
pub(crate) struct NativeDatabase {
    writer_endpoint: Endpoint,
    reader_endpoint: Option<Endpoint>,
    pools: Pools,
    events: Arc<dyn Observer>,
}

pyo3::create_exception!(
    _native,
    NativeDatabaseFailure,
    pyo3::exceptions::PyException,
    "A native database operation failed. The kind attribute is litellm-db's ErrorKind, and unavailable and requeue_safe are the error's is_unavailable and is_requeue_safe."
);

fn map_error(error: litellm_db::Error) -> PyErr {
    let kind: &'static str = error.kind().into();
    let failure = NativeDatabaseFailure::new_err(litellm_error::Report::new(&error).to_string());
    let decided = Python::attach(|py| {
        let value = failure.value(py);
        value.setattr("kind", kind)?;
        value.setattr("unavailable", error.is_unavailable())?;
        value.setattr("requeue_safe", error.is_requeue_safe())
    });
    match decided {
        Ok(()) => failure,
        Err(error) => error,
    }
}

#[pymethods]
impl NativeDatabase {
    #[staticmethod]
    #[pyo3(signature = (events, writer, reader=None))]
    fn connect(
        py: Python<'_>,
        events: Py<PyAny>,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] writer: Endpoint,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] reader: Option<Endpoint>,
    ) -> PyResult<Bound<'_, PyAny>> {
        crate::execution::run_async_value(py, async move {
            let pools = Pools::new(&writer, reader.as_ref()).map_err(map_error)?;
            Ok(NativeDatabase {
                writer_endpoint: writer,
                reader_endpoint: reader,
                pools,
                events: Arc::new(PythonEvents(events)),
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

    fn health_check<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let writer = self.pools.writer().clone();
        let events = Arc::clone(&self.events);
        let socket_timeout = self.writer_endpoint.socket_timeout().map_err(map_error)?;
        // Boxed so each future wrapper between here and the tokio worker holds a pointer instead of a copy of
        // the SQLx connect state, which overflowed the worker's 2 MiB stack in unoptimized builds
        crate::execution::run_async(
            py,
            Box::pin(async move {
                litellm_db::observe(
                    events.as_ref(),
                    HEALTH_CHECK,
                    litellm_db::within(socket_timeout, litellm_db::health_check(&writer)),
                )
                .await
            }),
            map_error,
        )
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
