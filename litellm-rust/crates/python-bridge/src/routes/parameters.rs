use std::collections::BTreeSet;

use litellm_core_utils::{call_arguments::CallArguments, params::is_control_param};
use litellm_host_python::{from_py, lookup};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde_json::{Map, Value};

/// Names whose `None` came from a signature default rather than the caller.
pub(super) fn defaulted_keys(
    bound: &Bound<'_, PyDict>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<BTreeSet<String>> {
    bound
        .iter()
        .filter(|(_, value)| value.is_none())
        .map(|(key, _)| {
            let name: String = key.extract()?;
            Ok((!kwargs.contains(&name)?).then_some(name))
        })
        .filter_map(PyResult::transpose)
        .collect()
}

pub(super) fn provider_parameters(
    py: Python<'_>,
    prepared: &Bound<'_, PyDict>,
    bound: &Bound<'_, PyDict>,
    defaulted: &BTreeSet<String>,
    inputs: &[&str],
    provider_owned: &[&str],
) -> PyResult<CallArguments> {
    let owned = py
        .import("litellm.types.utils")?
        .getattr("is_litellm_owned_kwarg")?;
    project_parameters(
        prepared,
        bound,
        defaulted,
        inputs,
        provider_owned,
        &|name| owned.call1((name,))?.extract(),
    )
}

fn project_parameters(
    prepared: &Bound<'_, PyDict>,
    bound: &Bound<'_, PyDict>,
    defaulted: &BTreeSet<String>,
    inputs: &[&str],
    provider_owned: &[&str],
    owned: &impl Fn(&str) -> PyResult<bool>,
) -> PyResult<CallArguments> {
    let is_provider_field = |name: &str| -> PyResult<bool> {
        if inputs.contains(&name) || is_control_param(name) {
            return Ok(false);
        }
        Ok(provider_owned.contains(&name) || !owned(name)?)
    };
    let names = bound
        .keys()
        .iter()
        .chain(prepared.keys().iter())
        .map(|key| key.extract::<String>())
        .collect::<PyResult<BTreeSet<_>>>()?;
    let fields = names
        .iter()
        .map(|name| provider_field(prepared, bound, defaulted, &is_provider_field, name))
        .filter_map(PyResult::transpose)
        .collect::<PyResult<Map<String, Value>>>()?;
    let overrides = lookup(prepared, bound.as_any(), "extra_body")?
        .filter(|value| !value.is_none())
        .map(|value| override_fields(&value, &is_provider_field))
        .transpose()?;
    let arguments: CallArguments = fields
        .into_iter()
        .chain(overrides.map(|fields| ("extra_body".to_string(), Value::Object(fields))))
        .collect();
    arguments
        .resolve_body_overrides()
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn provider_field(
    prepared: &Bound<'_, PyDict>,
    bound: &Bound<'_, PyDict>,
    defaulted: &BTreeSet<String>,
    is_provider_field: &impl Fn(&str) -> PyResult<bool>,
    name: &str,
) -> PyResult<Option<(String, Value)>> {
    if !is_provider_field(name)? {
        return Ok(None);
    }
    let Some(value) = lookup(prepared, bound.as_any(), name)? else {
        return Ok(None);
    };
    if value.is_none() && defaulted.contains(name) {
        return Ok(None);
    }
    Ok(Some((name.to_string(), from_py(&value)?)))
}

fn override_fields(
    extra_body: &Bound<'_, PyAny>,
    is_provider_field: &impl Fn(&str) -> PyResult<bool>,
) -> PyResult<Map<String, Value>> {
    let mapping = extra_body
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("extra_body must be an object"))?;
    mapping
        .iter()
        .map(|(key, value)| {
            let name: String = key.extract()?;
            if !is_provider_field(&name)? {
                return Ok(None);
            }
            Ok(Some((name, from_py(&value)?)))
        })
        .filter_map(PyResult::transpose)
        .collect()
}

#[cfg(test)]
mod tests {
    use litellm_host_python::to_py;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::null(json!(null))]
    #[case::false_value(json!(false))]
    #[case::zero(json!(0))]
    #[case::nested(json!({"mode": "new", "values": [true, null, {"nested": 7}]}))]
    #[case::json_schema(json!({"format": {"type": "json_schema", "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
    }}, "previous_message_id": null}))]
    fn unknown_fields_survive_projection(#[case] extension: Value) {
        Python::initialize();
        Python::attach(|py| {
            let source = json!({"future_provider_option": extension});
            let bound = to_py(py, &source).unwrap();
            let bound = bound.bind(py).cast::<PyDict>().unwrap();
            let parameters = project_parameters(
                &PyDict::new(py),
                bound,
                &defaulted_keys(bound, bound).unwrap(),
                &[],
                &[],
                &|_| Ok(false),
            )
            .unwrap();
            assert_eq!(serde_json::to_value(parameters).unwrap(), source);
        });
    }

    #[rstest]
    fn projection_resolves_prepared_values_and_overrides_without_serializing_controls() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
opaque = object()
kwargs = {'future_null': None, 'drop_params': True}
bound = {'model': 'resolved', 'messages': [], 'temperature': 0.25,
         'future_default': None, 'future_null': None, 'future_cleared': True,
         'drop_params': True, 'base_url': 'https://ignored.example',
         'metadata': {'user_id': 'keep'}, 'callbacks': [opaque], 'api_key': opaque}
prepared = {'future_cleared': None, 'temperature': 0.5,
            'extra_body': {'temperature': 0.75, 'future_override': None,
                           'model': 'ignored', 'messages': ['ignored'],
                           'litellm_metadata': opaque, 'callbacks': [opaque], 'api_key': opaque}}
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let dict = |name: &str| {
                locals
                    .get_item(name)
                    .unwrap()
                    .unwrap()
                    .cast_into::<PyDict>()
                    .unwrap()
            };
            let (kwargs, bound, prepared) = (dict("kwargs"), dict("bound"), dict("prepared"));
            let defaulted = defaulted_keys(&bound, &kwargs).unwrap();
            assert_eq!(defaulted, BTreeSet::from(["future_default".to_string()]));
            let parameters = project_parameters(
                &prepared,
                &bound,
                &defaulted,
                &["messages"],
                &["metadata"],
                &|name| Ok(name.starts_with("litellm_") || name == "metadata"),
            )
            .unwrap();
            assert_eq!(
                serde_json::to_value(parameters).unwrap(),
                json!({"temperature": 0.75, "future_null": null, "future_cleared": null,
                       "future_override": null, "metadata": {"user_id": "keep"}})
            );
            assert!(
                bound
                    .get_item("api_key")
                    .unwrap()
                    .unwrap()
                    .is(locals.get_item("opaque").unwrap().unwrap())
            );
        });
    }

    #[rstest]
    #[case::boolean(json!(false))]
    #[case::number(json!(0))]
    #[case::array(json!([]))]
    #[case::string(json!(""))]
    fn invalid_overrides_are_terminal(#[case] overrides: Value) {
        Python::initialize();
        Python::attach(|py| {
            let bound = to_py(py, &json!({"extra_body": overrides})).unwrap();
            let error = project_parameters(
                &PyDict::new(py),
                bound.bind(py).cast::<PyDict>().unwrap(),
                &BTreeSet::new(),
                &[],
                &[],
                &|_| Ok(false),
            )
            .unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }
}
