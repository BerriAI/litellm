use std::sync::Arc;

use litellm_core::integrations::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, LogError, LogFuture, ModelCallDetails,
};
use pyo3::prelude::*;
use serde_json::{json, Map, Value};

use super::callback_bridge::{call_python_awaitable, json_to_py, py_datetime_from_epoch_seconds};

fn dropped_log_error(message: impl Into<String>) -> LogError {
    LogError {
        message: message.into(),
        kind: "PythonCallbackError".to_string(),
    }
}

fn model_call_details_json(details: &ModelCallDetails) -> Value {
    let mut kwargs: Map<String, Value> = details.extra_metadata.clone().into_iter().collect();
    kwargs.insert("model".to_string(), json!(details.model));
    kwargs.insert(
        "custom_llm_provider".to_string(),
        json!(details.custom_llm_provider),
    );
    kwargs.insert(
        "call_type".to_string(),
        json!(details.call_type.to_string()),
    );
    kwargs.insert(
        "litellm_call_id".to_string(),
        json!(details.litellm_call_id),
    );
    kwargs.insert("request_id".to_string(), json!(details.request_id));
    kwargs.insert("response_cost".to_string(), json!(details.response_cost));
    kwargs.insert(
        "metadata".to_string(),
        json!({
            "user_api_key_hash": details.metadata.user_api_key_hash,
            "user_api_key_user_id": details.metadata.user_api_key_user_id,
            "user_api_key_team_id": details.metadata.user_api_key_team_id,
        }),
    );
    kwargs.insert(
        "standard_logging_object".to_string(),
        json!(details.standard_logging_payload),
    );
    if let Some(error) = &details.failure_error {
        kwargs.insert(
            "failure_error".to_string(),
            json!({
                "message": error.message,
                "kind": error.kind,
            }),
        );
    }
    Value::Object(kwargs)
}

pub struct PythonCustomLoggerAdapter {
    obj: Py<PyAny>,
}

impl PythonCustomLoggerAdapter {
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { obj }
    }
}

impl CustomLogger for PythonCustomLoggerAdapter {
    fn async_log_success_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        response_obj: &'a CallbackValue,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        let obj = Python::with_gil(|py| self.obj.clone_ref(py));
        let details = model_call_details_json(model_call_details);
        let response = json!({
            "object": response_obj.object,
            "value": response_obj.value,
        });
        Box::pin(async move {
            let args = Python::with_gil(|py| -> PyResult<Vec<Py<PyAny>>> {
                Ok(vec![
                    json_to_py(py, details)?,
                    json_to_py(py, response)?,
                    py_datetime_from_epoch_seconds(py, timing.start_time)?,
                    py_datetime_from_epoch_seconds(py, timing.end_time)?,
                ])
            })
            .map_err(|err| dropped_log_error(err.to_string()))?;
            call_python_awaitable(obj, "async_log_success_event", args, None)
                .await
                .map_err(|err| dropped_log_error(err.to_string()))?;
            Ok(())
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        response_obj: Option<&'a CallbackValue>,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        let obj = Python::with_gil(|py| self.obj.clone_ref(py));
        let details = model_call_details_json(model_call_details);
        let response = response_obj.map(|value| {
            json!({
                "object": value.object,
                "value": value.value,
            })
        });
        Box::pin(async move {
            let args = Python::with_gil(|py| -> PyResult<Vec<Py<PyAny>>> {
                Ok(vec![
                    json_to_py(py, details)?,
                    match response {
                        Some(response) => json_to_py(py, response)?,
                        None => py.None(),
                    },
                    py_datetime_from_epoch_seconds(py, timing.start_time)?,
                    py_datetime_from_epoch_seconds(py, timing.end_time)?,
                ])
            })
            .map_err(|err| dropped_log_error(err.to_string()))?;
            call_python_awaitable(obj, "async_log_failure_event", args, None)
                .await
                .map_err(|err| dropped_log_error(err.to_string()))?;
            Ok(())
        })
    }
}

pub fn py_callbacks_to_rust(
    py: Python<'_>,
    callbacks: Option<Py<PyAny>>,
) -> PyResult<Vec<Arc<dyn CustomLogger>>> {
    let Some(callbacks) = callbacks else {
        return Ok(Vec::new());
    };
    callbacks
        .bind(py)
        .try_iter()?
        .map(|callback| {
            let callback = callback?;
            Ok(Arc::new(PythonCustomLoggerAdapter::new(callback.unbind()))
                as Arc<dyn CustomLogger>)
        })
        .collect()
}
