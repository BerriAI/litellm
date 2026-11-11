use litellm_config::{GeneralSettings, LitellmRole, UserHeaderMapping};
use litellm_gateway_auth_core::end_user::{self, Body, EndUserSettings, IdValue};
use litellm_python_compat::{Value, json, repr};
use pyo3::{
    intern,
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString},
};
use serde::{
    Deserialize,
    de::{IntoDeserializer, value::Error as RoleError},
};

#[pyfunction]
pub(crate) fn auth_end_user_id(
    request_body: &Bound<'_, PyAny>,
    request_headers: Option<&Bound<'_, PyAny>>,
    general_settings: &Bound<'_, PyAny>,
    validate_end_user_id_in_db: bool,
) -> PyResult<Option<String>> {
    let headers = match request_headers {
        Some(headers) => headers
            .call_method0(intern!(headers.py(), "items"))?
            .try_iter()?
            .map(|item| {
                let (name, value) = item?.extract::<(String, Bound<'_, PyAny>)>()?;
                Ok((name, id_value(&value)?))
            })
            .collect::<PyResult<Vec<_>>>()?,
        None => Vec::new(),
    };
    let body = Body {
        user: id_value(&get(request_body, "user")?)?,
        litellm_metadata_user: metadata_id(&get(request_body, "litellm_metadata")?, "user")?,
        metadata_user_id: metadata_id(&get(request_body, "metadata")?, "user_id")?,
        safety_identifier: id_value(&get(request_body, "safety_identifier")?)?,
    };
    let general = GeneralSettings {
        user_header_name: get(general_settings, "user_header_name")?
            .cast::<PyString>()
            .ok()
            .map(|name| name.to_str().map(str::to_owned))
            .transpose()?,
        user_header_mappings: user_header_mappings(&get(
            general_settings,
            "user_header_mappings",
        )?)?,
        ..GeneralSettings::default()
    };
    let settings = EndUserSettings::new(&general, validate_end_user_id_in_db);
    Ok(end_user::resolve(&headers, &body, &settings))
}

fn get<'py>(mapping: &Bound<'py, PyAny>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    mapping.call_method1(intern!(mapping.py(), "get"), (key,))
}

fn id_value(value: &Bound<'_, PyAny>) -> PyResult<IdValue> {
    Ok(if value.is_instance_of::<PyBool>() {
        IdValue::Unusable
    } else if value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>() {
        IdValue::Number(value.str()?.to_str()?.to_owned())
    } else if let Ok(text) = value.cast::<PyString>() {
        IdValue::Text(text.to_str()?.to_owned())
    } else {
        IdValue::Unusable
    })
}

/// The id at `key` in a metadata dict, or in a metadata JSON string as `json.loads` reads it.
fn metadata_id(value: &Bound<'_, PyAny>, key: &str) -> PyResult<IdValue> {
    if let Ok(dict) = value.cast::<PyDict>() {
        return match dict.get_item(key)? {
            Some(value) => id_value(&value),
            None => Ok(IdValue::Unusable),
        };
    }
    match value.cast::<PyString>() {
        Ok(text) => Ok(json_metadata_id(text.to_str()?, key)),
        Err(_) => Ok(IdValue::Unusable),
    }
}

fn json_metadata_id(text: &str, key: &str) -> IdValue {
    let Ok(Value::Dict(entries)) = json::loads(text) else {
        return IdValue::Unusable;
    };
    match entries
        .iter()
        .rfind(|(name, _)| matches!(name, Value::Str(name) if name == key))
        .map(|(_, value)| value)
    {
        Some(Value::Str(text)) => IdValue::Text(text.clone()),
        Some(value @ (Value::Int(_) | Value::Float(_))) => IdValue::Number(repr::to_str(value)),
        _ => IdValue::Unusable,
    }
}

/// `user_header_mappings` as the proxy reads it: one mapping or a list, skipping entries that are
/// not dicts, lack a role or header name, or name a role this proxy does not have.
fn user_header_mappings(value: &Bound<'_, PyAny>) -> PyResult<Box<[UserHeaderMapping]>> {
    if !value.is_truthy()? {
        return Ok(Box::default());
    }
    let items: Vec<_> = match value.cast::<PyList>() {
        Ok(list) => list.iter().collect(),
        Err(_) => vec![value.clone()],
    };
    items
        .iter()
        .filter_map(|item| item.cast::<PyDict>().ok())
        .filter_map(|item| user_header_mapping(item).transpose())
        .collect()
}

fn user_header_mapping(item: &Bound<'_, PyDict>) -> PyResult<Option<UserHeaderMapping>> {
    let role = item
        .get_item("litellm_user_role")?
        .filter(|role| !role.is_none());
    let header_name = item.get_item("header_name")?;
    let (Some(role), Some(header_name)) = (role, header_name) else {
        return Ok(None);
    };
    if !header_name.is_truthy()? {
        return Ok(None);
    }
    let role: String = role
        .str()?
        .call_method0(intern!(item.py(), "lower"))?
        .extract()?;
    let role = LitellmRole::deserialize(IntoDeserializer::<RoleError>::into_deserializer(
        role.as_str(),
    ));
    Ok(match (role, header_name.extract::<String>()) {
        (Ok(litellm_user_role), Ok(header_name)) => Some(UserHeaderMapping {
            header_name,
            litellm_user_role,
        }),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use litellm_gateway_auth_core::end_user::IdValue;
    use rstest::rstest;

    use super::json_metadata_id;

    fn number(text: &str) -> IdValue {
        IdValue::Number(text.to_owned())
    }

    #[rstest]
    #[case::text(r#"{"user_id": "m"}"#, IdValue::Text("m".to_owned()))]
    #[case::other_key(r#"{"user": "m"}"#, IdValue::Unusable)]
    #[case::int(r#"{"user_id": 7}"#, number("7"))]
    #[case::float_as_str(r#"{"user_id": 1e16}"#, number("1e+16"))]
    #[case::big_int(r#"{"user_id": 100000000000000000000}"#, number("100000000000000000000"))]
    #[case::overflowing_float(r#"{"user_id": 1e400}"#, number("inf"))]
    #[case::bool(r#"{"user_id": true}"#, IdValue::Unusable)]
    #[case::repeated_key_takes_last(r#"{"user_id": "a", "user_id": "b"}"#, IdValue::Text("b".to_owned()))]
    #[case::array(r#"[{"user_id": "m"}]"#, IdValue::Unusable)]
    #[case::invalid_json("not json", IdValue::Unusable)]
    fn json_metadata_reads_like_json_loads(#[case] text: &str, #[case] expected: IdValue) {
        assert_eq!(json_metadata_id(text, "user_id"), expected);
    }
}
