use std::collections::HashMap;

use litellm_host_python::{from_py, to_py};
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{BoundMap, Held, Slot};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

/// The dict bound to `litellm.model_cost` for one install, under one GIL hold
pub(crate) struct BoundDict<'a, 'py> {
    dict: &'a Bound<'py, PyDict>,
    /// The row object each alias group shares in this install, as main's alias expansion gives them
    groups: HashMap<String, Py<PyDict>>,
}

impl<'a, 'py> BoundDict<'a, 'py> {
    pub(crate) fn new(dict: &'a Bound<'py, PyDict>) -> Self {
        Self {
            dict,
            groups: HashMap::new(),
        }
    }

    fn py(&self) -> Python<'py> {
        self.dict.py()
    }

    fn row(&self, key: &str) -> PyResult<Bound<'py, PyDict>> {
        let row = self
            .dict
            .get_item(key)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_owned()))?;
        Ok(row.cast_into::<PyDict>()?)
    }
}

fn held_value(value: &Bound<'_, PyAny>) -> Held {
    from_py::<Value>(value).map_or(Held::Foreign, Held::Json)
}

impl BoundMap for BoundDict<'_, '_> {
    type Error = PyErr;

    fn slot(&self, key: &str) -> PyResult<Slot> {
        Ok(match self.dict.get_item(key)? {
            None => Slot::Missing,
            Some(value) if value.is_instance_of::<PyDict>() => Slot::Row,
            Some(_) => Slot::NotARow,
        })
    }

    fn held(&self, key: &str, field: &str) -> PyResult<Held> {
        Ok(self
            .row(key)?
            .get_item(field)?
            .map_or(Held::Absent, |value| held_value(&value)))
    }

    fn fields(&self, key: &str) -> PyResult<Vec<(String, Held)>> {
        Ok(self
            .row(key)?
            .iter()
            .filter_map(|(field, value)| {
                Some((field.extract::<String>().ok()?, held_value(&value)))
            })
            .collect())
    }

    fn row_is_empty(&self, key: &str) -> PyResult<bool> {
        Ok(self.row(key)?.is_empty())
    }

    fn set_field(&mut self, key: &str, field: &str, value: &Value) -> PyResult<()> {
        self.row(key)?.set_item(field, to_py(self.py(), value)?)
    }

    fn del_field(&mut self, key: &str, field: &str) -> PyResult<()> {
        self.row(key)?.del_item(field)
    }

    fn insert_row(&mut self, key: &str, group: &str, fields: &Fields) -> PyResult<()> {
        let py = self.py();
        let row = match self.groups.get(group) {
            Some(row) => row.clone_ref(py),
            None => {
                let row = PyDict::new(py);
                for (field, value) in fields {
                    row.set_item(field, to_py(py, value)?)?;
                }
                let row = row.unbind();
                self.groups.insert(group.to_owned(), row.clone_ref(py));
                row
            }
        };
        self.dict.set_item(key, row)
    }

    fn del_key(&mut self, key: &str) -> PyResult<()> {
        self.dict.del_item(key)
    }

    fn retain_keys(&mut self, keep: &dyn Fn(&str) -> bool) -> PyResult<usize> {
        let stale: Vec<Bound<'_, PyAny>> = self
            .dict
            .keys()
            .iter()
            .filter(|key| key.extract::<String>().map_or(true, |key| !keep(&key)))
            .collect();
        let removed = stale.len();
        for key in stale {
            self.dict.del_item(key)?;
        }
        Ok(removed)
    }
}
