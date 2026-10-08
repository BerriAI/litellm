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
    let overrides = field(request, "extra_body")?
        .map(|extra_body| {
            let extra_body = extra_body
                .cast_into::<PyDict>()
                .map_err(|_| PyValueError::new_err("extra_body must be an object"))?;
            provider_fields(&extra_body, &is_provider_field, true)
        })
        .transpose()?;
    Ok(provider_fields(request, &is_provider_field, false)?
        .into_iter()
        .chain(overrides.into_iter().flatten())
        .collect())
}

fn provider_fields(
    fields: &Bound<'_, PyDict>,
    is_provider_field: &impl Fn(&str) -> PyResult<bool>,
    keep_none: bool,
) -> PyResult<Map<String, Value>> {
    fields
        .iter()
        .filter(|(_, value)| keep_none || !value.is_none())
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
    use std::ffi::CString;

    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn python_dict<'py>(py: Python<'py>, literal: &str) -> Bound<'py, PyDict> {
        py.eval(&CString::new(literal).unwrap(), None, None)
            .unwrap()
            .cast_into()
            .unwrap()
    }

    fn project(request: &Bound<'_, PyDict>) -> PyResult<Value> {
        let owned = |name: &str| Ok(name.starts_with("litellm_") || name == "metadata");
        project_parameters(request, "messages", &owned)
            .map(|parameters| serde_json::to_value(parameters).unwrap())
    }

    #[rstest]
    #[case::falsy_values_survive(
        "{'temperature': 0, 'logprobs': False, 'stop': '', 'tools': [], 'response_format': {}}",
        json!({"temperature": 0, "logprobs": false, "stop": "", "tools": [], "response_format": {}})
    )]
    #[case::nested_nulls_survive(
        "{'response_format': {'schema': None, 'enum': [None, 1]}}",
        json!({"response_format": {"schema": null, "enum": [null, 1]}})
    )]
    #[case::unknown_provider_fields_survive(
        "{'top_k': 40, 'thinking': {'type': 'enabled', 'budget_tokens': 1024}}",
        json!({"top_k": 40, "thinking": {"type": "enabled", "budget_tokens": 1024}})
    )]
    #[case::top_level_none_is_absent_for_known_names(
        "{'stream': None, 'temperature': None, 'top_p': 1}",
        json!({"top_p": 1})
    )]
    #[case::top_level_none_is_absent_for_unknown_names("{'future_option': None}", json!({}))]
    #[case::route_input_is_extracted_separately(
        "{'messages': [{'role': 'user'}], 'temperature': 1}",
        json!({"temperature": 1})
    )]
    #[case::another_routes_input_is_just_a_field("{'input': 'hi'}", json!({"input": "hi"}))]
    #[case::litellm_owned_names_are_dropped(
        "{'litellm_call_id': 'call', 'litellm_logging_obj': object(), 'temperature': 1}",
        json!({"temperature": 1})
    )]
    #[case::metadata_is_forwarded_although_owned(
        "{'metadata': {'user_id': 'u'}}",
        json!({"metadata": {"user_id": "u"}})
    )]
    #[case::opaque_controls_are_never_serialized(
        "{'api_key': object(), 'callbacks': [object()], 'extra_body': {'api_key': object()}}",
        json!({})
    )]
    #[case::extra_body_overrides_a_top_level_value(
        "{'temperature': 0.2, 'extra_body': {'temperature': 0.9}}",
        json!({"temperature": 0.9})
    )]
    #[case::extra_body_adds_fields("{'extra_body': {'top_k': 5}}", json!({"top_k": 5}))]
    #[case::extra_body_keeps_an_explicit_null(
        "{'temperature': 0.2, 'extra_body': {'temperature': None}}",
        json!({"temperature": null})
    )]
    #[case::extra_body_replaces_objects_whole(
        "{'thinking': {'type': 'enabled', 'budget_tokens': 1}, 'extra_body': {'thinking': {'type': 'disabled'}}}",
        json!({"thinking": {"type": "disabled"}})
    )]
    #[case::extra_body_none_is_absent("{'extra_body': None, 'top_p': 1}", json!({"top_p": 1}))]
    #[case::extra_body_empty_is_absent("{'extra_body': {}}", json!({}))]
    #[case::extra_body_cannot_smuggle_input_or_owned_names(
        "{'extra_body': {'messages': [], 'litellm_call_id': 'call', 'extra_body': {'top_k': 1}}}",
        json!({})
    )]
    #[case::extra_body_can_set_forwarded_metadata(
        "{'metadata': {'a': 1}, 'extra_body': {'metadata': {'b': 2}}}",
        json!({"metadata": {"b": 2}})
    )]
    fn projects_provider_fields(#[case] request: &str, #[case] expected: Value) {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(project(&python_dict(py, request)).unwrap(), expected);
        });
    }

    #[derive(Debug)]
    enum Placement {
        TopLevel,
        ExtraBody,
    }

    #[rstest]
    fn controls_never_reach_the_provider(
        #[values(
            "model",
            "api_key",
            "api_base",
            "base_url",
            "custom_llm_provider",
            "extra_headers",
            "timeout",
            "request_timeout",
            "max_retries",
            "callbacks",
            "drop_params",
            "additional_drop_params",
            "aws_secret_access_key",
            "vertex_credentials"
        )]
        name: &str,
        #[values(Placement::TopLevel, Placement::ExtraBody)] placement: Placement,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let control = PyDict::new(py);
            control.set_item(name, "secret").unwrap();
            let request = match placement {
                Placement::TopLevel => control,
                Placement::ExtraBody => {
                    let request = PyDict::new(py);
                    request.set_item("extra_body", control).unwrap();
                    request
                }
            };
            assert_eq!(project(&request).unwrap(), json!({}));
        });
    }

    #[rstest]
    #[case::extra_body_bool("{'extra_body': False}")]
    #[case::extra_body_number("{'extra_body': 0}")]
    #[case::extra_body_list("{'extra_body': []}")]
    #[case::extra_body_string("{'extra_body': ''}")]
    #[case::non_json_provider_value("{'top_k': object()}")]
    #[case::non_json_override_value("{'extra_body': {'top_k': object()}}")]
    #[case::non_string_name("{1: 'value'}")]
    fn invalid_provider_input_is_terminal(#[case] request: &str) {
        Python::initialize();
        Python::attach(|py| {
            assert!(project(&python_dict(py, request)).is_err());
        });
    }

    #[rstest]
    #[case::hook_rewrite_wins("{'temperature': 0.2}", "{'temperature': 0.5}", json!({"temperature": 0.5}))]
    #[case::hook_none_clears_a_bound_value("{'temperature': 0.2}", "{'temperature': None}", json!({}))]
    #[case::hook_adds_a_field("{}", "{'top_k': 5}", json!({"top_k": 5}))]
    #[case::bound_only_fields_survive("{'top_p': 1}", "{'temperature': 0.5}", json!({"top_p": 1, "temperature": 0.5}))]
    #[case::hook_extra_body_replaces_bound_extra_body(
        "{'extra_body': {'top_k': 1, 'seed': 2}}",
        "{'extra_body': {'top_k': 3}}",
        json!({"top_k": 3})
    )]
    fn merged_request_overlays_hook_rewrites(
        #[case] bound: &str,
        #[case] hooked: &str,
        #[case] expected: Value,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let bound = python_dict(py, bound);
            let before = bound.repr().unwrap().to_string();
            let request = merged_request(&bound, &python_dict(py, hooked)).unwrap();
            assert_eq!(project(&request).unwrap(), expected);
            assert_eq!(bound.repr().unwrap().to_string(), before);
        });
    }

    #[rstest]
    #[case::missing("{}", false)]
    #[case::none("{'model': None}", false)]
    #[case::empty_string("{'model': ''}", true)]
    #[case::zero("{'model': 0}", true)]
    #[case::false_value("{'model': False}", true)]
    fn field_treats_only_none_as_absent(#[case] request: &str, #[case] present: bool) {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                field(&python_dict(py, request), "model").unwrap().is_some(),
                present
            );
        });
    }
}
