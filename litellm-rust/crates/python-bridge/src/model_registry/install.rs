use std::collections::HashMap;

use indexmap::IndexMap;
use litellm_host_python::{from_py, to_py};
use litellm_model_catalog::Layers;
use litellm_model_catalog::wire::Fields;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

/// The field values litellm last wrote into one dict object
#[derive(Clone, Debug)]
pub(crate) struct Record {
    pub(crate) dict_id: usize,
    rows: HashMap<String, Fields>,
}

pub(crate) struct Outcome {
    pub(crate) record: Record,
    pub(crate) written: usize,
    pub(crate) removed: usize,
}

/// What the dict holds for one field, as litellm compares it
#[derive(PartialEq)]
enum Held {
    Absent,
    Json(Value),
    /// A value that does not convert to JSON, which only the user can have written
    Foreign,
}

/// One pass over the bound dict under one GIL hold. A field is written only while litellm still owns it, meaning the
/// dict holds the value last written there, or nothing where nothing was written. A dict that is not the object of the
/// last install is written in full, which is main's result after rebind then adopt
pub(crate) fn write_into(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    composed: &IndexMap<String, Fields>,
    layers: &Layers,
    previous: Option<Record>,
) -> PyResult<Outcome> {
    let dict_id = dict.as_ptr() as usize;
    let Some(previous) = previous else {
        return write_in_full(py, dict, composed, layers, dict_id);
    };
    let mut rows = previous.rows;
    let mut written = 0;
    let mut removed = 0;
    let mut fresh: HashMap<String, Py<PyDict>> = HashMap::new();
    let keys: Vec<String> = composed
        .keys()
        .chain(rows.keys().filter(|key| !composed.contains_key(*key)))
        .cloned()
        .collect();
    for key in keys {
        let target = composed.get(&key);
        let last = rows.remove(&key);
        let Some(current) = dict.get_item(&key)? else {
            match (target, last) {
                (Some(target), None) => {
                    dict.set_item(&key, shared_row(py, &mut fresh, layers, &key, target)?)?;
                    rows.insert(key, target.clone());
                    written += 1;
                }
                (_, Some(last)) if target.is_some() => {
                    rows.insert(key, last);
                }
                _ => {}
            }
            continue;
        };
        let Ok(row) = current.cast::<PyDict>() else {
            if let Some(last) = last {
                rows.insert(key, last);
            }
            continue;
        };
        let last = last.unwrap_or_default();
        let (owned, changed) = write_owned_fields(py, row, target, &last)?;
        if target.is_none() && changed && row.is_empty() {
            dict.del_item(&key)?;
            removed += 1;
            continue;
        }
        written += usize::from(changed);
        if !owned.is_empty() {
            rows.insert(key, owned);
        }
    }
    Ok(Outcome {
        record: Record { dict_id, rows },
        written,
        removed,
    })
}

fn write_in_full(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    composed: &IndexMap<String, Fields>,
    layers: &Layers,
    dict_id: usize,
) -> PyResult<Outcome> {
    let stale: Vec<Bound<'_, PyAny>> = dict
        .keys()
        .iter()
        .filter(|key| {
            key.extract::<String>()
                .map_or(true, |key| !composed.contains_key(&key))
        })
        .collect();
    let removed = stale.len();
    for key in stale {
        dict.del_item(key)?;
    }
    let mut fresh: HashMap<String, Py<PyDict>> = HashMap::new();
    let mut written = 0;
    for (key, target) in composed {
        let current = dict.get_item(key)?;
        let row = current.as_ref().and_then(|row| row.cast::<PyDict>().ok());
        match row {
            Some(row) => {
                let owned_by_litellm: Fields = row
                    .iter()
                    .filter_map(|(field, value)| {
                        let field = field.extract::<String>().ok()?;
                        Some((field, from_py::<Value>(&value).ok()?))
                    })
                    .collect();
                let extra: Vec<String> = row
                    .keys()
                    .iter()
                    .filter_map(|field| field.extract::<String>().ok())
                    .filter(|field| !owned_by_litellm.contains_key(field))
                    .collect();
                for field in extra {
                    row.del_item(field)?;
                }
                let (_, changed) = write_owned_fields(py, row, Some(target), &owned_by_litellm)?;
                written += usize::from(changed);
            }
            None => {
                dict.set_item(key, shared_row(py, &mut fresh, layers, key, target)?)?;
                written += 1;
            }
        }
    }
    Ok(Outcome {
        record: Record {
            dict_id,
            rows: composed
                .iter()
                .map(|(key, fields)| (key.clone(), fields.clone()))
                .collect(),
        },
        written,
        removed,
    })
}

/// Sets or removes each field litellm owns, and returns the owned fields as now written and whether any changed
fn write_owned_fields(
    py: Python<'_>,
    row: &Bound<'_, PyDict>,
    target: Option<&Fields>,
    last: &Fields,
) -> PyResult<(Fields, bool)> {
    let empty = Fields::new();
    let target = target.unwrap_or(&empty);
    let fields: Vec<&String> = target
        .keys()
        .chain(last.keys().filter(|field| !target.contains_key(*field)))
        .collect();
    let mut owned = Fields::new();
    let mut changed = false;
    for field in fields {
        let held = held(row, field)?;
        let litellm_owns = match (&held, last.get(field)) {
            (Held::Absent, None) => true,
            (Held::Json(value), Some(last)) => value == last,
            _ => false,
        };
        if !litellm_owns {
            if let Some(last) = last.get(field) {
                owned.insert(field.clone(), last.clone());
            }
            continue;
        }
        match target.get(field) {
            Some(value) => {
                if held != Held::Json(value.clone()) {
                    row.set_item(field, to_py(py, value)?)?;
                    changed = true;
                }
                owned.insert(field.clone(), value.clone());
            }
            None if held != Held::Absent => {
                row.del_item(field)?;
                changed = true;
            }
            None => {}
        }
    }
    Ok((owned, changed))
}

fn held(row: &Bound<'_, PyDict>, field: &str) -> PyResult<Held> {
    Ok(match row.get_item(field)? {
        None => Held::Absent,
        Some(value) => from_py::<Value>(&value).map_or(Held::Foreign, Held::Json),
    })
}

/// A new row, the same object for every key of one alias group, as main's alias expansion gives them
fn shared_row(
    py: Python<'_>,
    fresh: &mut HashMap<String, Py<PyDict>>,
    layers: &Layers,
    key: &str,
    fields: &Fields,
) -> PyResult<Py<PyDict>> {
    let owner = layers.alias_owner(key).unwrap_or(key).to_owned();
    if let Some(row) = fresh.get(&owner) {
        return Ok(row.clone_ref(py));
    }
    let row = PyDict::new(py);
    for (field, value) in fields {
        row.set_item(field, to_py(py, value)?)?;
    }
    let row = row.unbind();
    fresh.insert(owner, row.clone_ref(py));
    Ok(row)
}
