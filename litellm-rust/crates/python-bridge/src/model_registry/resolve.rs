use std::borrow::Cow;
use std::sync::{Arc, Mutex, PoisonError};

use litellm_host_python::{from_py, to_py};
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{
    Candidates, Catalog, ImageCandidates, Keys, Resolution, RowRef, model_info, rejects_row,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use serde_json::Value;

use super::name_index::NameIndex;
use crate::cost::PricesHandle;

const CANDIDATES_MODULE: &str = "litellm.rust_bridge.host.model_names";

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Resolved {
    #[pyo3(get)]
    key: String,
    /// The row as it was when `resolve` ran
    fields: Arc<Fields>,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Missing {
    #[pyo3(get)]
    tried: Vec<String>,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Invalid {
    #[pyo3(get)]
    key: String,
    #[pyo3(get)]
    errors: Vec<String>,
    fields: Arc<Fields>,
}

#[pymethods]
impl Resolved {
    #[pyo3(signature = (provider = None))]
    fn model_info(&self, py: Python<'_>, provider: Option<&str>) -> PyResult<Py<PyAny>> {
        to_py(py, &model_info(&self.key, &self.fields, provider))
    }

    fn supports(&self, capability: &str) -> bool {
        self.fields.get(capability) == Some(&Value::Bool(true))
    }

    /// The row as `resolve` read it, for the Python steps that still merge into it
    fn row(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, self.fields.as_ref())
    }

    /// The row's typed prices and uplifts. `resolve` already reported a row whose prices do not lower as `Invalid`
    #[getter]
    fn prices(&self) -> PricesHandle {
        PricesHandle::lower(&self.key, &self.fields).unwrap_or_default()
    }
}

impl Invalid {
    pub(crate) fn new(key: String, errors: Vec<String>, fields: Arc<Fields>) -> Self {
        Self {
            key,
            errors,
            fields,
        }
    }
}

#[pymethods]
impl Invalid {
    #[pyo3(signature = (provider = None))]
    fn model_info(&self, py: Python<'_>, provider: Option<&str>) -> PyResult<Py<PyAny>> {
        to_py(py, &model_info(&self.key, &self.fields, provider))
    }

    fn row(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, self.fields.as_ref())
    }
}

/// The live map as `resolve` sees it, the dict bound when the call started (decision 1)
pub(crate) struct LiveKeys<'a, 'py> {
    pub(crate) dict: &'a Bound<'py, PyDict>,
    pub(crate) index: &'a Mutex<Option<NameIndex>>,
}

impl LiveKeys<'_, '_> {
    fn indexed<T>(&self, read: impl Fn(&NameIndex) -> T) -> T {
        let mut index = self.index.lock().unwrap_or_else(PoisonError::into_inner);
        if !index
            .as_ref()
            .is_some_and(|index| index.is_current(self.dict))
        {
            *index = Some(NameIndex::build(self.dict));
        }
        read(index.as_ref().expect("the index was just built"))
    }

    fn rebuild(&self) {
        *self.index.lock().unwrap_or_else(PoisonError::into_inner) =
            Some(NameIndex::build(self.dict));
    }

    pub(crate) fn provider_keys(&self, provider: &str) -> Vec<String> {
        self.indexed(|index| index.provider_keys(provider).to_vec())
    }

    /// The row's fields that convert to JSON
    pub(crate) fn fields_of(&self, key: &str) -> Option<Fields> {
        self.fields(key).map(|(fields, _)| fields)
    }

    /// Every field that converts to JSON. A value only Python can hold is left out, and reported by `full_row`
    fn fields(&self, key: &str) -> Option<(Fields, bool)> {
        let row = self.dict.get_item(key).ok().flatten()?;
        let row = row.cast::<PyDict>().ok()?;
        if let Ok(fields) = from_py::<Fields>(row) {
            return Some((fields, true));
        }
        let fields = row
            .iter()
            .filter_map(|(name, value)| {
                Some((
                    name.extract::<String>().ok()?,
                    from_py::<Value>(&value).ok()?,
                ))
            })
            .collect();
        Some((fields, false))
    }
}

impl Keys for LiveKeys<'_, '_> {
    fn contains(&self, key: &str) -> bool {
        self.dict.contains(key).unwrap_or(false)
    }

    fn row(&self, key: &str) -> Option<RowRef<'_>> {
        self.fields(key).map(|(fields, _)| Cow::Owned(fields))
    }

    fn by_normalized(&self, name: &str) -> Option<String> {
        let hit = self.indexed(|index| index.by_lower(name).map(str::to_owned))?;
        if self.contains(&hit) {
            return Some(hit);
        }
        self.rebuild();
        self.indexed(|index| index.by_lower(name).map(str::to_owned))
            .filter(|key| self.contains(key))
    }
}

/// `get_model_info`'s lookup. Python builds the candidate names, Rust walks them over the live dict
pub(crate) fn resolve(
    py: Python<'_>,
    keys: &LiveKeys<'_, '_>,
    model: &str,
    provider: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let (candidates, provider) = candidates(py, model, provider)?;
    match litellm_model_catalog::resolve(keys, &candidates, provider.as_deref()) {
        Resolution::Missing { tried } => {
            Ok(Missing { tried }.into_pyobject(py)?.into_any().unbind())
        }
        Resolution::Found { key, .. } => found(py, keys, key),
    }
}

/// The row at `key` as `Resolved`, or `Invalid` when it fails validation
pub(crate) fn found(py: Python<'_>, keys: &LiveKeys<'_, '_>, key: String) -> PyResult<Py<PyAny>> {
    let (fields, converted) = keys.fields(&key).unwrap_or_default();
    let errors: Vec<String> = if converted {
        Catalog::row_errors(&key, &fields)
            .iter()
            .filter(|error| rejects_row(&error.error))
            .map(ToString::to_string)
            .collect()
    } else {
        vec![format!("{key} must contain JSON-compatible values")]
    };
    let fields = Arc::new(fields);
    if errors.is_empty() {
        Ok(Resolved { key, fields }
            .into_pyobject(py)?
            .into_any()
            .unbind())
    } else {
        Ok(Invalid {
            key,
            errors,
            fields,
        }
        .into_pyobject(py)?
        .into_any()
        .unbind())
    }
}

/// main's `default_image_cost_calculator` lookup, the first of its eight key spellings in the live dict
pub(crate) fn resolve_image(
    py: Python<'_>,
    keys: &LiveKeys<'_, '_>,
    candidates: &ImageCandidates,
) -> PyResult<Py<PyAny>> {
    let tried = candidates.spellings();
    match tried.iter().find(|key| keys.contains(key)) {
        Some(key) => found(py, keys, key.clone()),
        None => Ok(Missing { tried }.into_pyobject(py)?.into_any().unbind()),
    }
}

fn candidates(
    py: Python<'_>,
    model: &str,
    provider: Option<&str>,
) -> PyResult<(Candidates, Option<String>)> {
    let names = py
        .import(CANDIDATES_MODULE)?
        .call_method1("candidates", (model, provider))?;
    let names = names.cast::<PyTuple>()?;
    let name = |index: usize| -> PyResult<Option<String>> { names.get_item(index)?.extract() };
    Ok((
        Candidates {
            combined_model_name: name(0)?,
            model: model.to_owned(),
            region_free_combined_model_name: name(1)?,
            split_model: name(2)?,
            combined_stripped_model_name: name(3)?,
            stripped_model_name: name(4)?,
            provider_prefixed_model_name: name(5)?,
        },
        name(6)?,
    ))
}
