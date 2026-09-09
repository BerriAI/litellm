use std::sync::Arc;

use futures_util::StreamExt;
use litellm_core::integrations::custom_logger::{LogError, LogFuture};
use litellm_core::lifecycle::{
    CallLifecycleContext, Clock, RouteProjection, TerminalClassification, TerminalDispatcher,
    TerminalRecord,
};
use litellm_core::messages::types::ProviderMessagesRequest;
use litellm_python_interop::AsyncByteStream;
use pyo3::prelude::*;

use crate::errors::messages_provider_error_to_pyerr;

struct Session {
    host: Py<PyAny>,
    locals: pyo3_async_runtimes::TaskLocals,
    drain: litellm_core::lifecycle::StreamDrainPolicy,
}

impl litellm_core::lifecycle::StreamDrain for Session {
    fn stream_drain_policy(&self) -> litellm_core::lifecycle::StreamDrainPolicy {
        self.drain.clone()
    }
}

impl Clock for Session {
    fn now(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs_f64())
            .unwrap_or_default()
    }
}

impl TerminalDispatcher for Session {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        Box::pin(async move {
            let result = Python::attach(|py| {
                let value = match &terminal.projection {
                    RouteProjection::Messages { value } => value,
                    _ => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "invalid Messages terminal projection",
                        ));
                    }
                };
                let error = match &terminal.classification {
                    TerminalClassification::Success | TerminalClassification::Incomplete { .. } => {
                        None
                    }
                    TerminalClassification::Cancelled { message } => {
                        Some(format!("Cancelled: {message}"))
                    }
                    TerminalClassification::Failure { kind, message } => {
                        Some(format!("{kind}: {message}"))
                    }
                };
                let value = litellm_python_interop::to_py(py, value)?;
                let coroutine = self
                    .host
                    .bind(py)
                    .call_method1("complete_stream", (value, error, terminal.timing.end_time))?;
                pyo3_async_runtimes::into_future_with_locals(&self.locals, coroutine)
            });
            match result {
                Ok(future) => future.await.map(|_| ()).map_err(log_error),
                Err(error) => Err(log_error(error)),
            }
        })
    }
}

fn log_error(error: PyErr) -> LogError {
    LogError {
        kind: "PythonCallback".into(),
        message: error.to_string(),
    }
}

pub(super) fn send(
    py: Python<'_>,
    permit: litellm_core::lifecycle::program::ProviderPermit<
        litellm_core::messages::lifecycle::MessagesRoute,
    >,
    request: ProviderMessagesRequest,
    host: Py<PyAny>,
) -> PyResult<Bound<'_, PyAny>> {
    let arguments = host.bind(py).getattr("current")?;
    let context = CallLifecycleContext::new(
        "messages",
        arguments.get_item("model")?.extract::<String>()?,
        arguments
            .call_method1("get", ("custom_llm_provider", "anthropic"))?
            .extract::<String>()?,
        arguments
            .call_method1("get", ("litellm_call_id", ""))?
            .extract::<String>()?,
    );
    let start_time = host
        .bind(py)
        .getattr("start")?
        .call_method0("timestamp")?
        .extract::<f64>()?;
    static DRAIN: std::sync::OnceLock<litellm_core::lifecycle::StreamDrainPolicy> =
        std::sync::OnceLock::new();
    let max_detached = py
        .import("litellm.constants")?
        .getattr("ANTHROPIC_MESSAGES_MAX_DETACHED_STREAM_DRAINS")?
        .extract::<usize>()?;
    let drain = DRAIN.get_or_init(|| {
        litellm_core::lifecycle::StreamDrainPolicy::with_max_detached(max_detached)
    });
    let drain = request
        .timeout()
        .map_or_else(|| drain.clone(), |timeout| drain.with_idle_timeout(timeout));
    let services = Arc::new(Session {
        drain,
        host,
        locals: pyo3_async_runtimes::tokio::get_current_locals(py)?,
    });
    litellm_python_interop::run_async_py(py, async move {
        let call = permit
            .messages_stream(request, context, start_time, services)
            .await
            .map_err(messages_provider_error_to_pyerr)?;
        let completion = call.completion.register();
        let stream = call
            .stream
            .map(|item| item.map_err(messages_provider_error_to_pyerr));
        Ok(AsyncByteStream::new(
            Box::pin(stream),
            Box::pin(async move {
                completion
                    .await
                    .map(|_| ())
                    .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
            }),
        ))
    })
}
