use std::sync::Arc;

use litellm_core::integrations::custom_guardrail::{
    CustomGuardrail, GuardrailContext, GuardrailDecision, GuardrailError, GuardrailEventHook,
    GuardrailFuture, GuardrailRequest,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

use super::callback_bridge::{call_python_awaitable, json_to_py, py_attr_string, py_to_json};

fn py_guardrail_name(py: Python<'_>, obj: &Py<PyAny>) -> String {
    py_attr_string(py, obj, "guardrail_name").unwrap_or_else(|| {
        obj.bind(py)
            .get_type()
            .name()
            .map(|name| name.to_string())
            .unwrap_or_else(|_| "python_guardrail".to_string())
    })
}

fn guardrail_event_hook_from_str(value: &str) -> Option<GuardrailEventHook> {
    match value {
        "pre_call" | "GuardrailEventHooks.pre_call" => Some(GuardrailEventHook::PreCall),
        "during_call" | "GuardrailEventHooks.during_call" => Some(GuardrailEventHook::DuringCall),
        _ => None,
    }
}

fn py_guardrail_hooks(py: Python<'_>, obj: &Py<PyAny>) -> Vec<GuardrailEventHook> {
    let Some(attr) = obj.bind(py).getattr("event_hook").ok() else {
        return vec![GuardrailEventHook::PreCall, GuardrailEventHook::DuringCall];
    };
    if attr.is_none() {
        return vec![GuardrailEventHook::PreCall, GuardrailEventHook::DuringCall];
    }
    if let Ok(list) = attr.downcast::<PyList>() {
        return list
            .iter()
            .filter_map(|item| {
                let value = item
                    .getattr("value")
                    .ok()
                    .and_then(|value_attr| value_attr.extract::<String>().ok())
                    .or_else(|| item.extract::<String>().ok())
                    .or_else(|| item.str().ok().map(|value| value.to_string()))?;
                guardrail_event_hook_from_str(&value)
            })
            .collect();
    }
    let value = attr
        .getattr("value")
        .ok()
        .and_then(|value_attr| value_attr.extract::<String>().ok())
        .or_else(|| attr.extract::<String>().ok())
        .or_else(|| attr.str().ok().map(|value| value.to_string()));
    value
        .as_deref()
        .and_then(guardrail_event_hook_from_str)
        .map(|hook| vec![hook])
        .unwrap_or_default()
}

pub struct PythonCustomGuardrailAdapter {
    obj: Py<PyAny>,
    name: String,
    hooks: Vec<GuardrailEventHook>,
}

impl PythonCustomGuardrailAdapter {
    pub fn new(py: Python<'_>, obj: Py<PyAny>) -> Self {
        let name = py_guardrail_name(py, &obj);
        let hooks = py_guardrail_hooks(py, &obj);
        Self { obj, name, hooks }
    }

    fn call_guardrail_hook<'a>(
        &'a self,
        method_name: &'static str,
        context: &'a GuardrailContext,
        request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        let obj = Python::with_gil(|py| self.obj.clone_ref(py));
        let call_type = context.call_type.to_string();
        Box::pin(async move {
            let (kwargs, data) = Python::with_gil(|py| -> PyResult<(Py<PyDict>, Py<PyAny>)> {
                let kwargs = PyDict::new(py);
                kwargs.set_item("user_api_key_dict", py.None())?;
                kwargs.set_item("cache", py.None())?;
                let data = json_to_py(py, request.data)?;
                kwargs.set_item("data", data.bind(py))?;
                kwargs.set_item("call_type", call_type)?;
                Ok((kwargs.unbind(), data))
            })
            .map_err(|err| GuardrailError::blocked(err.to_string()))?;
            let result = call_python_awaitable(obj, method_name, Vec::new(), Some(kwargs))
                .await
                .map_err(|err| GuardrailError::blocked(err.to_string()))?;
            Python::with_gil(|py| -> Result<GuardrailDecision, GuardrailError> {
                let result = result.bind(py);
                if result.is_none() {
                    let value = py_to_json(py, data.bind(py))
                        .map_err(|err| GuardrailError::blocked(err.to_string()))?;
                    return Ok(GuardrailDecision::Allow(GuardrailRequest::new(value)));
                }
                if let Ok(message) = result.extract::<String>() {
                    return Err(GuardrailError::blocked(message));
                }
                let value = py_to_json(py, result)
                    .map_err(|err| GuardrailError::blocked(err.to_string()))?;
                if value.is_object() {
                    Ok(GuardrailDecision::Mask(GuardrailRequest::new(value)))
                } else {
                    let fallback = py_to_json(py, data.bind(py))
                        .map_err(|err| GuardrailError::blocked(err.to_string()))?;
                    Ok(GuardrailDecision::Mask(GuardrailRequest::new(fallback)))
                }
            })
        })
    }
}

impl CustomGuardrail for PythonCustomGuardrailAdapter {
    fn guardrail_name(&self) -> &str {
        &self.name
    }

    fn supported_event_hooks(&self) -> &[GuardrailEventHook] {
        &self.hooks
    }

    fn async_pre_call_hook<'a>(
        &'a self,
        context: &'a GuardrailContext,
        request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        self.call_guardrail_hook("async_pre_call_hook", context, request)
    }

    fn async_moderation_hook<'a>(
        &'a self,
        context: &'a GuardrailContext,
        request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        self.call_guardrail_hook("async_moderation_hook", context, request)
    }
}

pub fn py_guardrails_to_rust(
    py: Python<'_>,
    guardrails: Option<Py<PyAny>>,
) -> PyResult<Vec<Arc<dyn CustomGuardrail>>> {
    let Some(guardrails) = guardrails else {
        return Ok(Vec::new());
    };
    guardrails
        .bind(py)
        .try_iter()?
        .map(|guardrail| {
            let guardrail = guardrail?;
            Ok(
                Arc::new(PythonCustomGuardrailAdapter::new(py, guardrail.unbind()))
                    as Arc<dyn CustomGuardrail>,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_supported_python_guardrail_hooks() {
        assert_eq!(
            guardrail_event_hook_from_str("pre_call"),
            Some(GuardrailEventHook::PreCall)
        );
        assert_eq!(
            guardrail_event_hook_from_str("GuardrailEventHooks.during_call"),
            Some(GuardrailEventHook::DuringCall)
        );
    }

    #[test]
    fn leaves_unsupported_python_guardrail_hooks_unmapped() {
        assert_eq!(guardrail_event_hook_from_str("post_call"), None);
        assert_eq!(guardrail_event_hook_from_str("logging_only"), None);
        assert_eq!(guardrail_event_hook_from_str("pre_mcp_call"), None);
        assert_eq!(guardrail_event_hook_from_str("during_mcp_call"), None);
        assert_eq!(
            guardrail_event_hook_from_str("realtime_input_transcription"),
            None
        );
    }
}
