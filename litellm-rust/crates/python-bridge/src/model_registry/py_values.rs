use litellm_host_python::{from_py, to_py};
use litellm_model_catalog::wire::Fields;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use serde_json::{Map, Value};

const MARKER: &str = "\u{0}litellm:python-object:";

/// Python values a registration carries that JSON cannot, such as a deployment's `created_at` datetime. Each one stands
/// in the merged row as a marker string and is put back when the row is written
#[derive(Default)]
pub(crate) struct Opaque(Vec<Py<PyAny>>);

pub(crate) fn is_opaque(value: &Value) -> bool {
    value.as_str().is_some_and(|text| text.starts_with(MARKER))
}

impl Opaque {
    pub(crate) fn fields(&mut self, row: &Bound<'_, PyAny>) -> PyResult<Fields> {
        let Value::Object(fields) = self.value(row)? else {
            return Ok(Fields::new());
        };
        Ok(fields)
    }

    fn value(&mut self, value: &Bound<'_, PyAny>) -> PyResult<Value> {
        if let Ok(dict) = value.cast::<PyDict>() {
            let mut fields = Map::new();
            for (key, item) in dict.iter() {
                let Ok(key) = key.cast::<PyString>().map(|key| key.to_string()) else {
                    continue;
                };
                fields.insert(key, self.value(&item)?);
            }
            return Ok(Value::Object(fields));
        }
        if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
            return value
                .try_iter()?
                .map(|item| self.value(&item?))
                .collect::<PyResult<Vec<_>>>()
                .map(Value::Array);
        }
        if let Ok(json) = from_py::<Value>(value) {
            return Ok(json);
        }
        self.0.push(value.clone().unbind());
        Ok(Value::String(format!("{MARKER}{}", self.0.len() - 1)))
    }

    pub(crate) fn to_py(&self, py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
        match value {
            Value::Object(fields) => {
                let dict = PyDict::new(py);
                for (key, item) in fields {
                    dict.set_item(key, self.to_py(py, item)?)?;
                }
                Ok(dict.into_any().unbind())
            }
            Value::Array(items) => {
                let list = PyList::empty(py);
                for item in items {
                    list.append(self.to_py(py, item)?)?;
                }
                Ok(list.into_any().unbind())
            }
            Value::String(text) => match text
                .strip_prefix(MARKER)
                .and_then(|index| index.parse::<usize>().ok())
            {
                Some(index) => Ok(self
                    .0
                    .get(index)
                    .map_or_else(|| py.None(), |object| object.clone_ref(py))),
                None => to_py(py, value),
            },
            value => to_py(py, value),
        }
    }
}

/// A persisted registration outlives the call, and its Python-only values with it would not survive a replay
pub(crate) fn without_opaque(fields: &Fields) -> Fields {
    fields
        .iter()
        .filter(|(_, value)| !is_opaque(value))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}
