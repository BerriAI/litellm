use litellm_core_utils::{call_arguments::CallArguments, params::is_control_param};
use litellm_host_python::from_py;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde_json::{Map, Value};

const PROVIDER_FORWARDED: &[&str] = &["metadata"];

pub(super) fn merged_request<'py>(
    bound: &Bound<'py, PyDict>,
    hooked: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let request = bound.copy()?;
    request.update(hooked.as_mapping())?;
    Ok(request)
}

pub(super) fn field<'py>(
    request: &Bound<'py, PyDict>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(request.get_item(name)?.filter(|value| !value.is_none()))
}

pub(super) fn provider_parameters(
    py: Python<'_>,
    request: &Bound<'_, PyDict>,
    input: &str,
) -> PyResult<CallArguments> {
    let owned = py
        .import("litellm.types.utils")?
        .getattr("is_litellm_owned_kwarg")?;
    project_parameters(request, input, &|name| owned.call1((name,))?.extract())
}

fn project_parameters(
    request: &Bound<'_, PyDict>,
    input: &str,
    owned: &impl Fn(&str) -> PyResult<bool>,
) -> PyResult<CallArguments> {
    let is_provider_field = |name: &str| -> PyResult<bool> {
        if name == input || is_control_param(name) {
            return Ok(false);
        }
        Ok(PROVIDER_FORWARDED.contains(&name) || !owned(name)?)
    };
    let fields = request
        .iter()
        .filter(|(_, value)| !value.is_none())
        .map(|(key, value)| {
            let name: String = key.extract()?;
            if !is_provider_field(&name)? {
                return Ok(None);
            }
            Ok(Some((name, from_py(&value)?)))
        })
        .filter_map(PyResult::transpose)
        .collect::<PyResult<Map<String, Value>>>()?;
    let overrides = field(request, "extra_body")?
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

    fn dict<'py>(py: Python<'py>, source: &Value) -> Bound<'py, PyDict> {
        to_py(py, source)
            .unwrap()
            .into_bound(py)
            .cast_into()
            .unwrap()
    }

    #[rstest]
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
            let parameters =
                project_parameters(&dict(py, &source), "messages", &|_| Ok(false)).unwrap();
            assert_eq!(serde_json::to_value(parameters).unwrap(), source);
        });
    }

    #[rstest]
    #[case::stream("stream")]
    #[case::temperature("temperature")]
    #[case::unknown("future_provider_option")]
    fn a_top_level_none_is_absent_whatever_its_name(#[case] name: &str) {
        Python::initialize();
        Python::attach(|py| {
            let source = json!({name: null, "kept": 1});
            let parameters =
                project_parameters(&dict(py, &source), "messages", &|_| Ok(false)).unwrap();
            assert_eq!(
                serde_json::to_value(parameters).unwrap(),
                json!({"kept": 1})
            );
        });
    }

    #[test]
    fn hook_rewrites_win_over_bound_values() {
        Python::initialize();
        Python::attach(|py| {
            let bound = dict(
                py,
                &json!({"temperature": 0.25, "top_p": 0.9, "future_cleared": true, "stream": null}),
            );
            let hooked = dict(py, &json!({"temperature": 0.5, "future_cleared": null}));
            let request = merged_request(&bound, &hooked).unwrap();
            let parameters = project_parameters(&request, "messages", &|_| Ok(false)).unwrap();
            assert_eq!(
                serde_json::to_value(parameters).unwrap(),
                json!({"temperature": 0.5, "top_p": 0.9})
            );
            assert!(field(&request, "future_cleared").unwrap().is_none());
            assert_eq!(bound.len(), 4);
        });
    }

    #[test]
    fn projection_resolves_overrides_without_serializing_controls() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
opaque = object()
request = {'model': 'resolved', 'messages': [], 'temperature': 0.25,
           'drop_params': True, 'base_url': 'https://ignored.example',
           'metadata': {'user_id': 'keep'}, 'litellm_trace_id': 'owned',
           'callbacks': [opaque], 'api_key': opaque,
           'extra_body': {'temperature': 0.75, 'future_override': None,
                          'model': 'ignored', 'messages': ['ignored'],
                          'litellm_metadata': opaque, 'callbacks': [opaque], 'api_key': opaque}}
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let request = locals
                .get_item("request")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let parameters = project_parameters(&request, "messages", &|name| {
                Ok(name.starts_with("litellm_") || name == "metadata")
            })
            .unwrap();
            assert_eq!(
                serde_json::to_value(parameters).unwrap(),
                json!({"temperature": 0.75, "future_override": null, "metadata": {"user_id": "keep"}})
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
            let request = dict(py, &json!({"extra_body": overrides}));
            let error = project_parameters(&request, "messages", &|_| Ok(false)).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }
}
