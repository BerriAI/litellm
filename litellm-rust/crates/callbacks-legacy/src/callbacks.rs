//! Callback fan-out over litellm's `Logging` object: which callbacks are registered,
//! the deferred and worker-submitted success paths, and the sync-callbacks-for-async-calls
//! duplication. All of it expires with the legacy callback contract.

use litellm_callbacks::event::{RequestContext, WireRequest};
use litellm_host_python::to_py;
use pyo3::{exceptions::PyBaseException, prelude::*, types::PyDict};

use crate::logger::PythonLogger;

pub trait LegacyCallbacks {
    fn callbacks_needed(&self, py: Python<'_>, phase: &str) -> PyResult<bool>;

    /// `Logging.update_from_kwargs`: what the logger is told about the request it is
    /// about to see, with consumed credentials redacted.
    fn update_from_kwargs(
        &self,
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        wire: &WireRequest,
        context: &RequestContext,
    ) -> PyResult<()>;

    fn record_api_call_start(&self, py: Python<'_>) -> PyResult<()>;

    /// `Logging.pre_call`, or its payload-free shortcut when no input callback listens.
    fn pre_call(
        &self,
        py: Python<'_>,
        input: &str,
        api_key: Option<&Bound<'_, PyAny>>,
        body: &Bound<'_, PyDict>,
        headers: &Bound<'_, PyDict>,
        url: &str,
    ) -> PyResult<()>;

    /// `Logging.post_call`, or its payload-free shortcut when no input callback listens.
    fn post_call(
        &self,
        py: Python<'_>,
        original_response: &str,
        body: Option<&Py<PyDict>>,
        headers: Option<&Py<PyDict>>,
    ) -> PyResult<()>;

    fn defers_async_logging(&self, py: Python<'_>) -> bool;

    fn defer_success(&self, py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<()>;

    fn sync_success_for_async_call(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()>;

    fn failure(
        &self,
        py: Python<'_>,
        error: &Py<PyBaseException>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
        asynchronous: bool,
    ) -> PyResult<Option<Py<PyAny>>>;

    fn submit_success(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()>;

    fn enqueue_success(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()>;
}

impl LegacyCallbacks for PythonLogger {
    fn callbacks_needed(&self, py: Python<'_>, phase: &str) -> PyResult<bool> {
        if !self.bridge_owned() {
            return Ok(true);
        }
        py.import("litellm.rust_bridge.legacy_callbacks")?
            .getattr("callbacks_needed")?
            .call1((self.object(py), phase))?
            .extract()
    }

    fn update_from_kwargs(
        &self,
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        wire: &WireRequest,
        context: &RequestContext,
    ) -> PyResult<()> {
        let secret_fields: Vec<&str> = context.secret_fields.iter().map(String::as_str).collect();
        let update = PyDict::new(py);
        update.set_item("kwargs", redact(py, kwargs.bind(py), &secret_fields)?)?;
        update.set_item("model", &context.model)?;
        update.set_item(
            "optional_params",
            redact(
                py,
                &to_py(py, &context.optional_params)?
                    .into_bound(py)
                    .cast_into::<PyDict>()?,
                &secret_fields,
            )?,
        )?;
        let params = PyDict::new(py);
        params.set_item(
            "litellm_call_id",
            kwargs.bind(py).get_item("litellm_call_id")?,
        )?;
        params.set_item("api_base", &wire.url)?;
        for name in ["logger_fn", "litellm_request_debug"] {
            if let Some(value) = kwargs.bind(py).get_item(name)? {
                params.set_item(name, value)?;
            }
        }
        for name in custom_pricing_fields(py)? {
            if let Some(value) = kwargs.bind(py).get_item(&name)?
                && !value.is_none()
            {
                params.set_item(name, value)?;
            }
        }
        update.set_item("litellm_params", params)?;
        update.set_item("custom_llm_provider", &context.custom_llm_provider)?;
        self.object(py)
            .call_method("update_from_kwargs", (), Some(&update))?;
        Ok(())
    }

    fn record_api_call_start(&self, py: Python<'_>) -> PyResult<()> {
        self.object(py).call_method0("record_api_call_start_time")?;
        Ok(())
    }

    fn pre_call(
        &self,
        py: Python<'_>,
        input: &str,
        api_key: Option<&Bound<'_, PyAny>>,
        body: &Bound<'_, PyDict>,
        headers: &Bound<'_, PyDict>,
        url: &str,
    ) -> PyResult<()> {
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", body)?;
        additional.set_item("headers", headers)?;
        additional.set_item("api_base", url)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("input", input)?;
        kwargs.set_item("api_key", api_key)?;
        kwargs.set_item("additional_args", &additional)?;
        if self.callbacks_needed(py, "input")? {
            self.object(py).call_method("pre_call", (), Some(&kwargs))?;
        } else {
            self.object(py)
                .call_method("_pre_call", (), Some(&kwargs))?;
            self.record_api_call_start(py)?;
        }
        Ok(())
    }

    fn post_call(
        &self,
        py: Python<'_>,
        original_response: &str,
        body: Option<&Py<PyDict>>,
        headers: Option<&Py<PyDict>>,
    ) -> PyResult<()> {
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", body)?;
        additional.set_item("headers", headers)?;
        if self.callbacks_needed(py, "input")? {
            let kwargs = PyDict::new(py);
            kwargs.set_item("original_response", original_response)?;
            kwargs.set_item("additional_args", &additional)?;
            self.object(py)
                .call_method("post_call", (), Some(&kwargs))?;
        } else {
            let response = py
                .import("json")?
                .call_method1("dumps", (original_response,))?;
            self.object(py).call_method1(
                "record_post_call",
                (response, py.None(), py.None(), additional),
            )?;
        }
        Ok(())
    }
    fn defers_async_logging(&self, py: Python<'_>) -> bool {
        self.object(py)
            .getattr("_defer_async_logging")
            .is_ok_and(|value| value.is_truthy().unwrap_or(false))
    }

    fn defer_success(&self, py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<()> {
        self.object(py).setattr("_native_pending_logging", pending)
    }

    fn sync_success_for_async_call(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()> {
        if !self.callbacks_needed(py, "sync_success_async")? {
            return Ok(());
        }
        self.object(py).call_method1(
            "handle_sync_success_callbacks_for_async_calls",
            (response, start, end),
        )?;
        Ok(())
    }

    fn failure(
        &self,
        py: Python<'_>,
        error: &Py<PyBaseException>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
        asynchronous: bool,
    ) -> PyResult<Option<Py<PyAny>>> {
        if !self.callbacks_needed(
            py,
            if asynchronous {
                "async_failure"
            } else {
                "sync_failure"
            },
        )? {
            py.import("litellm.rust_bridge.legacy_callbacks")?
                .getattr("failure_bookkeeping")?
                .call1((self.object(py), error, start, end, asynchronous))?;
            return Ok(None);
        }
        let trace = py
            .import("traceback")?
            .getattr("format_exception")?
            .call1((error,))?;
        let trace = pyo3::types::PyString::new(py, "").call_method1("join", (trace,))?;
        let value = self.object(py).call_method1(
            if asynchronous {
                "async_failure_handler"
            } else {
                "failure_handler"
            },
            (error, trace, start, end),
        )?;
        Ok(asynchronous.then(|| value.unbind()))
    }
    fn submit_success(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()> {
        if !self.callbacks_needed(py, "sync_success")? {
            return self.success_bookkeeping(py, response, start, end, false);
        }
        let context = py.import("contextvars")?.call_method0("copy_context")?;
        py.import("litellm.litellm_core_utils.litellm_logging")?
            .getattr("executor")?
            .call_method1(
                "submit",
                (
                    context.getattr("run")?,
                    self.object(py).getattr("success_handler")?,
                    response,
                    start,
                    end,
                ),
            )?;
        Ok(())
    }

    fn enqueue_success(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()> {
        if !self.callbacks_needed(py, "async_success")? {
            return self.success_bookkeeping(py, response, start, end, true);
        }
        let context = py.import("contextvars")?.call_method0("copy_context")?;
        let worker = py
            .import("litellm.litellm_core_utils.logging_worker")?
            .getattr("GLOBAL_LOGGING_WORKER")?
            .getattr("ensure_initialized_and_enqueue")?;
        let coroutine = self
            .object(py)
            .call_method1("async_success_handler", (response, start, end))?;
        let enqueue = context.call_method1("run", (worker, &coroutine));
        if enqueue.is_err()
            && let Err(error) = coroutine.call_method0("close")
        {
            error.write_unraisable(py, Some(&coroutine));
        }
        enqueue.map(|_| ())
    }
}

fn custom_pricing_fields(py: Python<'_>) -> PyResult<Vec<String>> {
    py.import("litellm.types.utils")?
        .getattr("CustomPricingLiteLLMParams")?
        .getattr("model_fields")?
        .cast_into::<PyDict>()?
        .keys()
        .iter()
        .map(|name| name.extract::<String>())
        .collect()
}

fn redact(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    secret_fields: &[&str],
) -> PyResult<Py<PyDict>> {
    let redacted = PyDict::new(py);
    for (name, value) in params {
        let name = name.extract::<String>()?;
        if name == "proxy_server_request" {
            continue;
        }
        if secret_fields.contains(&name.as_str()) {
            redacted.set_item(name, "****")?;
        } else {
            redacted.set_item(name, value)?;
        }
    }
    Ok(redacted.unbind())
}

/// Proxy-internal calls skip the legacy success fan-out.
pub fn is_internal_call(py: Python<'_>) -> PyResult<bool> {
    py.import("litellm._internal_context")?
        .getattr("is_internal_call")?
        .call_method0("get")?
        .extract()
}

#[cfg(test)]
mod tests {
    use pyo3::types::PyDict;

    use super::*;

    fn logger_whose_registries_need_no_input(py: Python<'_>, bridge_owned: bool) -> PythonLogger {
        let locals = PyDict::new(py);
        py.run(
            c"
import sys
import types
for name in ('litellm', 'litellm.rust_bridge', 'litellm.rust_bridge.legacy_callbacks'):
    sys.modules.setdefault(name, types.ModuleType(name))
legacy = sys.modules['litellm.rust_bridge.legacy_callbacks']
legacy.callbacks_needed = lambda logger, phase: logger.needed.get(phase, True)
class Logger:
    needed = {'input': False}
logger = Logger()
",
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        PythonLogger::new(
            locals.get_item("logger").unwrap().unwrap().unbind(),
            bridge_owned,
        )
    }

    #[test]
    fn a_caller_owned_logger_is_observed_in_full() {
        Python::initialize();
        Python::attach(|py| {
            let logger = logger_whose_registries_need_no_input(py, false);
            assert!(logger.callbacks_needed(py, "input").unwrap());
        });
    }

    #[test]
    fn a_bridge_owned_logger_is_elided_where_no_registry_needs_it() {
        Python::initialize();
        Python::attach(|py| {
            let logger = logger_whose_registries_need_no_input(py, true);
            assert!(!logger.callbacks_needed(py, "input").unwrap());
            assert!(logger.callbacks_needed(py, "payload").unwrap());
        });
    }
}
