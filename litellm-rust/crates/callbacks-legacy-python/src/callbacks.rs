//! Callback fan-out over litellm's `Logging` object: which callbacks are registered,
//! the deferred and worker-submitted success paths, and the sync-callbacks-for-async-calls
//! duplication. All of it expires with the legacy callback contract.

use litellm_host::interceptors::{RequestContext, WireRequest};
use litellm_host_python::to_py;
use pyo3::{exceptions::PyBaseException, prelude::*, types::PyDict};
use serde_json::Value;

use crate::logger::PythonLogger;
use crate::python::{Logging, Wrapper};

pub trait LegacyCallbacks {
    /// `Logging.update_from_kwargs`: what the logger is told about the request it is
    /// about to see, with consumed credentials redacted.
    fn update_from_kwargs(
        &self,
        py: Python<'_>,
        kwargs: &Bound<'_, PyDict>,
        wire: &WireRequest,
        context: &RequestContext,
        request_params: Option<&serde_json::Map<String, Value>>,
    ) -> PyResult<()>;

    /// `Logging.pre_call`.
    fn pre_call(
        &self,
        py: Python<'_>,
        input: &str,
        api_key: Option<&str>,
        body: &Bound<'_, PyDict>,
        headers: &Bound<'_, PyDict>,
        url: &str,
    ) -> PyResult<()>;

    /// Records the provider's raw HTTP response on the logger, as Python's Messages handler
    /// does with `httpx_response` and `response_headers`.
    fn record_response(
        &self,
        py: Python<'_>,
        status: u16,
        headers: &[(String, String)],
        body: &str,
    ) -> PyResult<()>;

    /// Records one provider HTTP exchange's window, as Python's timed `AsyncHTTPHandler.post`
    /// does with `llm_api_duration_ms` and the metadata's `llm_api_timing_windows`.
    fn record_api_window(&self, py: Python<'_>, start: f64, end: f64) -> PyResult<()>;

    /// `Logging.post_call`.
    fn post_call(
        &self,
        py: Python<'_>,
        original_response: &str,
        api_key: Option<&str>,
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
    fn update_from_kwargs(
        &self,
        py: Python<'_>,
        kwargs: &Bound<'_, PyDict>,
        wire: &WireRequest,
        context: &RequestContext,
        request_params: Option<&serde_json::Map<String, Value>>,
    ) -> PyResult<()> {
        let secret_fields: Vec<&str> = context.secret_fields.iter().map(String::as_str).collect();
        let redacted_kwargs = redact(py, kwargs, &secret_fields)?;
        let optional_params = redact(
            py,
            &to_py(py, &context.optional_params)?
                .into_bound(py)
                .cast_into::<PyDict>()?,
            &secret_fields,
        )?;
        let params = PyDict::new(py);
        params.set_item("litellm_call_id", kwargs.get_item("litellm_call_id")?)?;
        params.set_item("api_base", &wire.url)?;
        for name in ["logger_fn", "litellm_request_debug"] {
            if let Some(value) = kwargs.get_item(name)? {
                params.set_item(name, value)?;
            }
        }
        for name in custom_pricing_fields(py)? {
            if let Some(value) = kwargs.get_item(&name)?
                && !value.is_none()
            {
                params.set_item(name, value)?;
            }
        }
        for (name, value) in request_params.into_iter().flatten() {
            params.set_item(name, to_py(py, value)?)?;
        }
        Logging::Update.call(
            py,
            (
                self.object(py),
                redacted_kwargs,
                &context.model,
                optional_params,
                params,
                &context.custom_llm_provider,
            ),
        )?;
        Ok(())
    }

    fn pre_call(
        &self,
        py: Python<'_>,
        input: &str,
        api_key: Option<&str>,
        body: &Bound<'_, PyDict>,
        headers: &Bound<'_, PyDict>,
        url: &str,
    ) -> PyResult<()> {
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", body)?;
        additional.set_item("headers", headers)?;
        additional.set_item("api_base", url)?;
        Logging::PreCall.call(py, (self.object(py), input, api_key, &additional))?;
        Ok(())
    }

    fn record_response(
        &self,
        py: Python<'_>,
        status: u16,
        headers: &[(String, String)],
        body: &str,
    ) -> PyResult<()> {
        Logging::RecordResponse.call(py, (self.object(py), status, headers.to_vec(), body))?;
        Ok(())
    }

    fn record_api_window(&self, py: Python<'_>, start: f64, end: f64) -> PyResult<()> {
        Logging::RecordApiWindow.call(py, (self.object(py), start, end))?;
        Ok(())
    }

    fn post_call(
        &self,
        py: Python<'_>,
        original_response: &str,
        api_key: Option<&str>,
        body: Option<&Py<PyDict>>,
        headers: Option<&Py<PyDict>>,
    ) -> PyResult<()> {
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", body)?;
        additional.set_item("headers", headers)?;
        Logging::PostCall.call(
            py,
            (self.object(py), original_response, api_key, &additional),
        )?;
        Ok(())
    }

    fn defers_async_logging(&self, py: Python<'_>) -> bool {
        Logging::DefersAsync
            .call(py, (self.object(py),))
            .and_then(|value| value.extract())
            .unwrap_or(false)
    }

    fn defer_success(&self, py: Python<'_>, pending: &Bound<'_, PyAny>) -> PyResult<()> {
        Logging::DeferSuccess.call(py, (self.object(py), pending))?;
        Ok(())
    }

    fn sync_success_for_async_call(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()> {
        Logging::SyncSuccessForAsyncCall.call(py, (self.object(py), response, start, end))?;
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
        let value =
            Logging::FailureHandler.call(py, (self.object(py), error, start, end, asynchronous))?;
        Ok(asynchronous.then(|| value.unbind()))
    }

    fn submit_success(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()> {
        Logging::SubmitSuccess.call(py, (self.object(py), response, start, end))?;
        Ok(())
    }

    fn enqueue_success(
        &self,
        py: Python<'_>,
        response: &Option<Py<PyAny>>,
        start: &Py<PyAny>,
        end: &Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let coroutine =
            Logging::AsyncSuccessHandler.call(py, (self.object(py), response, start, end))?;
        let enqueue = Logging::Enqueue.call(py, (&coroutine,));
        if enqueue.is_err()
            && let Err(error) = coroutine.call_method0("close")
        {
            error.write_unraisable(py, Some(&coroutine));
        }
        enqueue.map(|_| ())
    }
}

fn custom_pricing_fields(py: Python<'_>) -> PyResult<Vec<String>> {
    Logging::CustomPricingFields.call(py, ())?.extract()
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
    Wrapper::IsInternalCall.call(py, ())?.extract()
}
