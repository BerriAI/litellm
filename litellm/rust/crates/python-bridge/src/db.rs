use std::{sync::Arc, time::UNIX_EPOCH};

use litellm_db::{Call, ClientIdentity, Endpoint, Observer, Outcome, Pools, QueryEvent};
use litellm_db_resolve::{Limits, Overlay, REPLICA_URL, Rule, WRITER_URL};
use litellm_host_python::PythonContext;
use litellm_secret_value::SecretValue;
use pyo3::{exceptions::PyValueError, prelude::*};

/// Prisma's own default, which the pool takes when the url pins no `pool_timeout`.
const PRISMA_POOL_TIMEOUT_SECONDS: f64 = 10.0;

const HEALTH_CHECK: Call = Call {
    call_type: "health_check",
    table: None,
    operation: None,
};

tokio::task_local! {
    static CALLER: PythonContext;
}

/// Runs `operation` as the Python caller's database work: each query it reports is recorded inside the caller's
/// `contextvars` context, so service logging parents the event under the caller's request span.
pub(crate) fn for_caller<F: Future>(
    py: Python<'_>,
    operation: F,
) -> PyResult<impl Future<Output = F::Output> + use<F>> {
    Ok(CALLER.scope(PythonContext::capture(py)?, operation))
}

struct PythonEvents(Arc<Py<PyAny>>);

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
        let args = (
            event.call.call_type,
            event.call.table,
            event.call.operation,
            started_at,
            event.duration.as_secs_f64(),
            failure,
        );
        let events = Arc::clone(&self.0);
        let caller = CALLER.try_with(PythonContext::clone).ok();
        Python::attach(|py| {
            let recorded = match caller {
                Some(context) => {
                    let events = Arc::clone(&events);
                    context
                        .enter(py, move |py| {
                            events.bind(py).call_method1("record", args).map(|_| ())
                        })
                        .and_then(|recorded| recorded)
                }
                None => events.bind(py).call_method1("record", args).map(|_| ()),
            };
            if let Err(error) = recorded {
                error.write_unraisable(py, Some(events.bind(py)));
            }
        });
    }
}

/// The native pool's endpoint for the url Prisma connects with.
#[pyclass(frozen, name = "DatabaseEndpoint")]
pub(crate) struct DatabaseEndpoint(Endpoint);

#[pymethods]
impl DatabaseEndpoint {
    /// `max_connections` is the Rust share of the connection budget, whatever the url's `connection_limit`.
    #[staticmethod]
    #[pyo3(signature = (url, max_connections, *, replica=false, client_identity=None))]
    fn resolve(
        py: Python<'_>,
        url: String,
        max_connections: u32,
        replica: bool,
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] client_identity: Option<
            ClientIdentity,
        >,
    ) -> PyResult<Bound<'_, PyAny>> {
        let source = if replica { REPLICA_URL } else { WRITER_URL };
        let overlay = Overlay {
            limits: Limits {
                max_connections: Rule::OverUrl(max_connections),
                acquire_timeout: Rule::UnderUrl(Some(PRISMA_POOL_TIMEOUT_SECONDS)),
                idle_timeout: Rule::UnderUrl(None),
                socket_timeout: Rule::UnderUrl(None),
            },
            unnamed_statements: false,
            session_timeouts: Vec::new(),
            client_identity,
        };
        crate::execution::run_async_value(py, async move {
            let (endpoint, ignored) = litellm_db_resolve::interpret(source, &url, &overlay)
                .await
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            for ignored in &ignored {
                tracing::warn!(
                    "Database setting {} is ignored by the native pool: {}",
                    ignored.setting,
                    ignored.consequence
                );
            }
            Ok(DatabaseEndpoint(endpoint))
        })
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

pub(crate) fn map_error(error: litellm_db::Error) -> PyErr {
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

impl NativeDatabase {
    /// What a Rust repository needs to run its queries on these pools. Statements are named only when both
    /// sides name them, since a read can land on either.
    pub(crate) fn repository_database(&self) -> PyResult<litellm_repositories::Database> {
        let names_statements = self.writer_endpoint.names_statements()
            && self
                .reader_endpoint
                .as_ref()
                .is_none_or(Endpoint::names_statements);
        Ok(litellm_repositories::Database::new(
            self.pools.clone(),
            Arc::clone(&self.events),
            self.writer_endpoint.socket_timeout().map_err(map_error)?,
            names_statements,
        ))
    }
}

#[pymethods]
impl NativeDatabase {
    #[staticmethod]
    #[pyo3(signature = (events, writer, reader=None))]
    fn connect<'py>(
        py: Python<'py>,
        events: Py<PyAny>,
        writer: PyRef<'_, DatabaseEndpoint>,
        reader: Option<PyRef<'_, DatabaseEndpoint>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let writer = writer.0.clone();
        let reader = reader.map(|reader| reader.0.clone());
        crate::execution::run_async_value(py, async move {
            let pools = Pools::new(&writer, reader.as_ref()).map_err(map_error)?;
            Ok(NativeDatabase {
                writer_endpoint: writer,
                reader_endpoint: reader,
                pools,
                events: Arc::new(PythonEvents(Arc::new(events))),
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
