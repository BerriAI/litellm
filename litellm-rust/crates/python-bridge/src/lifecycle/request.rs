use litellm_core::call_lifecycle::provider::ProviderOptions;
use litellm_python_interop::from_py_preserving_errors as from_py;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Map, Value};

use super::contract::RequestField;

pub(crate) fn required<'py>(
    request: &Bound<'py, PyDict>,
    field: RequestField,
) -> PyResult<Bound<'py, PyAny>> {
    let name = field.as_str();
    request
        .get_item(field.key(request.py()))?
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err(format!("missing {name}")))
}

pub(crate) fn optional_string(
    request: &Bound<'_, PyDict>,
    field: RequestField,
) -> PyResult<Option<String>> {
    request
        .get_item(field.key(request.py()))?
        .map(|value| value.extract())
        .transpose()
        .map(Option::flatten)
}

pub(crate) fn object(
    request: &Bound<'_, PyDict>,
    field: RequestField,
) -> PyResult<Map<String, Value>> {
    request
        .get_item(field.key(request.py()))?
        .filter(|value| !value.is_none())
        .map(|value| from_py(&value))
        .transpose()
        .map(Option::unwrap_or_default)
}

pub(crate) fn options(request: &Bound<'_, PyDict>) -> PyResult<ProviderOptions> {
    Ok(ProviderOptions {
        model: required(request, RequestField::Model)?.extract()?,
        litellm_call_id: optional_string(request, RequestField::LitellmCallId)?,
        api_key: optional_string(request, RequestField::ApiKey)?,
        api_base: optional_string(request, RequestField::ApiBase)?,
        custom_llm_provider: optional_string(request, RequestField::CustomLlmProvider)?,
        extra_headers: request
            .get_item(RequestField::ExtraHeaders.key(request.py()))?
            .filter(|value| !value.is_none())
            .map(|value| from_py(&value))
            .transpose()?,
        timeout: crate::marshal::optional_timeout(
            request
                .get_item(RequestField::Timeout.key(request.py()))?
                .filter(|value| !value.is_none())
                .map(|value| crate::marshal::python_timeout_seconds(request.py(), value.unbind()))
                .transpose()?
                .flatten(),
        ),
    })
}
