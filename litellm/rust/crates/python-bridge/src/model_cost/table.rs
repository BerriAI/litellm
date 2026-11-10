//! Builds `litellm.model_cost` from an installed document

use std::borrow::Cow;
use std::collections::HashMap;

use indexmap::IndexSet;
use litellm_host_python::to_py;
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{CostMap, FALLBACK_GENERALIZATIONS, Written};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use super::host::{self, Landing};

const ALIASES: &str = "aliases";

enum Served<'a> {
    Model,
    AliasOf(&'a str),
    /// A top-level value that is not a model row, such as `sample_spec`
    Section(&'a Value),
}

/// Every served key in dict order. An `alias_of` row repeats among the aliases and keeps its document position
fn served(cost_map: &CostMap) -> impl Iterator<Item = (&str, Served<'_>)> {
    let catalog = cost_map.catalog();
    let documented = catalog
        .written()
        .filter(|(key, _)| *key != FALLBACK_GENERALIZATIONS)
        .filter_map(|(key, written)| {
            let served = match written {
                Written::Section(value) => Served::Section(value),
                Written::Entry(_) => match catalog.aliases().get(key) {
                    Some(owner) => Served::AliasOf(owner),
                    None => catalog.get(key).map(|_| Served::Model)?,
                },
            };
            Some((key, served))
        });
    let listed = cost_map.alias_groups().flat_map(|(owner, names)| {
        names
            .iter()
            .skip(1)
            .map(move |alias| (alias.as_str(), Served::AliasOf(owner)))
    });
    documented.chain(listed)
}

/// The installed document as a new dict
pub(crate) fn build<'py>(py: Python<'py>, cost_map: &CostMap) -> PyResult<Bound<'py, PyDict>> {
    let table = PyDict::new(py);
    let mut rows = HashMap::new();
    for (key, served) in served(cost_map) {
        let value = match served {
            Served::Model => row(py, cost_map, key, &mut rows)?,
            Served::AliasOf(owner) => row(py, cost_map, owner, &mut rows)?,
            Served::Section(value) => to_py(py, value)?.into_bound(py),
        };
        table.set_item(key, value)?;
    }
    Ok(table)
}

fn row<'a, 'py>(
    py: Python<'py>,
    cost_map: &'a CostMap,
    owner: &'a str,
    rows: &mut HashMap<&'a str, Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(row) = rows.get(owner) {
        return Ok(row.clone());
    }
    let fields = cost_map
        .catalog()
        .get(owner)
        .expect("a served model or alias owner is a catalog row");
    let fields = to_py(py, without_alias_list(fields).as_ref())?.into_bound(py);
    let row = match cost_map.names(owner) {
        [] => fields,
        names => host::entry(fields, names)?,
    };
    rows.insert(owner, row.clone());
    Ok(row)
}

fn without_alias_list(fields: &Fields) -> Cow<'_, Fields> {
    if fields.get(ALIASES).is_none_or(Value::is_null) {
        return Cow::Borrowed(fields);
    }
    Cow::Owned(
        fields
            .iter()
            .filter(|(name, _)| *name != ALIASES)
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
    )
}

/// The installed row under `key` with `applied` written over it in order, `None` when neither has one
pub(crate) fn registered_row<'py>(
    py: Python<'py>,
    cost_map: &CostMap,
    key: &str,
    applied: &[Bound<'py, PyDict>],
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let catalog = cost_map.catalog();
    let installed = catalog
        .get(key)
        .or_else(|| match catalog.section(key) {
            Some(Value::Object(fields)) if key != FALLBACK_GENERALIZATIONS => Some(fields),
            _ => None,
        })
        .map(without_alias_list);
    let row = match (installed, applied) {
        (None, []) => return Ok(None),
        (Some(fields), _) => to_py(py, fields.as_ref())?
            .into_bound(py)
            .cast_into::<PyDict>()?,
        (None, _) => PyDict::new(py),
    };
    for registration in applied {
        row.update(registration.as_mapping())?;
    }
    Ok(Some(row))
}

/// The keys `registered_row` answers for, installed keys first
pub(crate) fn registered_keys<'a>(
    cost_map: &'a CostMap,
    registered: impl Iterator<Item = &'a str>,
) -> Vec<String> {
    let served: IndexSet<&str> = served(cost_map).map(|(key, _)| key).collect();
    let only_registered: Vec<&str> = registered.filter(|key| !served.contains(key)).collect();
    served
        .into_iter()
        .chain(only_registered)
        .map(str::to_owned)
        .collect()
}

/// Writes `registration` over the row under `landing.key`, and under every name that shares that row object
pub(crate) fn write<'py>(
    rows: &Bound<'py, PyDict>,
    landing: &Landing<'py>,
    registration: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let current = rows.get_item(&landing.key)?;
    let row = host::merged(current.as_ref(), landing, registration)?;
    let names: Vec<String> = row.getattr("names")?.extract()?;
    let sharing: Vec<&String> = names
        .iter()
        .filter(|name| {
            let served = rows.get_item(name.as_str()).ok().flatten();
            served
                .zip(current.as_ref())
                .is_some_and(|(served, current)| served.is(current))
        })
        .collect();
    let written = PyDict::new(rows.py());
    written.set_item(&landing.key, &row)?;
    sharing
        .into_iter()
        .try_for_each(|name| written.set_item(name, &row))?;
    host::write(rows, &written)
}
