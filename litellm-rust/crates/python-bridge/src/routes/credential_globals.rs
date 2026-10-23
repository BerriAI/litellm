use litellm_auth::Sources;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::{
    coercion::FieldSpec,
    python_settings::{PythonSettings, Snapshot},
};

/// The settings among `sources` that have a `litellm.*` global and that `params` leaves unset,
/// counting a blank value as unset the way Python's `or` chain does.
pub(crate) fn unset<'s>(
    params: &Map<String, Value>,
    sources: impl IntoIterator<Item = &'s Sources>,
) -> Vec<&'s Sources> {
    sources
        .into_iter()
        .filter(|sources| sources.litellm_setting.is_some())
        .filter(|sources| {
            !sources
                .litellm_params
                .iter()
                .any(|name| params.get(*name).is_some_and(is_set))
        })
        .collect()
}

fn is_set(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(text) => !text.trim().is_empty(),
        _ => true,
    }
}

/// Reads the globals of `unset` through the `ProviderDefaults` accessor and places each truthy
/// one under its setting's first param name, replacing a blank value the call passed.
pub(crate) fn fold(
    py: Python<'_>,
    params: Map<String, Value>,
    unset: &[&Sources],
) -> PyResult<Map<String, Value>> {
    if unset.is_empty() {
        return Ok(params);
    }
    let Some(defaults) = PythonSettings::ProviderDefaults.read_or_unset(py)? else {
        return Ok(params);
    };
    fold_from(params, unset, &defaults)
}

pub(crate) fn fold_from(
    params: Map<String, Value>,
    unset: &[&Sources],
    defaults: &Snapshot<'_>,
) -> PyResult<Map<String, Value>> {
    let globals = unset
        .iter()
        .filter_map(|sources| {
            let setting = sources.litellm_setting?;
            let name = sources.litellm_params.first()?;
            let spec = FieldSpec::new(setting, |field| field.falsy_optional_string());
            Some(
                defaults
                    .read(&spec)
                    .map(|value| value.map(|value| (name.to_string(), Value::String(value)))),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(params
        .into_iter()
        .chain(globals.into_iter().flatten())
        .collect())
}
