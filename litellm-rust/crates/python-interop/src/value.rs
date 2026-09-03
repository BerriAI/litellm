use std::panic::{AssertUnwindSafe, catch_unwind};

use rustc_hash::FxHashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use serde_json::Value;

use crate::marshal::panic_to_pyerr;

pub struct PythonValue(pub Value);

impl<'py> IntoPyObject<'py> for PythonValue {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let mut keys = FxHashMap::default();
        catch_unwind(AssertUnwindSafe(|| convert(py, &self.0, &mut keys)))
            .map_err(panic_to_pyerr)?
    }
}

pub fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    let mut keys = FxHashMap::default();
    convert(py, value, &mut keys).map(Bound::unbind)
}

fn convert<'py, 'value>(
    py: Python<'py>,
    value: &'value Value,
    keys: &mut FxHashMap<&'value str, Py<PyString>>,
) -> PyResult<Bound<'py, PyAny>> {
    let converted = match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(boolean) => (*boolean).into_pyobject(py)?.to_owned().into_any(),
        Value::Number(number) => {
            if let Some(signed) = number.as_i64() {
                signed.into_pyobject(py)?.into_any()
            } else if let Some(unsigned) = number.as_u64() {
                unsigned.into_pyobject(py)?.into_any()
            } else {
                let float = number.as_f64().ok_or_else(|| {
                    PyValueError::new_err(format!("unsupported number: {number}"))
                })?;
                float.into_pyobject(py)?.into_any()
            }
        }
        Value::String(text) => PyString::new(py, text).into_any(),
        Value::Array(items) => {
            let mut elements = Vec::with_capacity(items.len());
            for item in items {
                elements.push(convert(py, item, keys)?);
            }
            PyList::new(py, elements)?.into_any()
        }
        Value::Object(entries) => {
            let dict = PyDict::new(py);
            for (name, item) in entries {
                let converted = convert(py, item, keys)?;
                let key = keys
                    .entry(name.as_str())
                    .or_insert_with(|| PyString::new(py, name).unbind());
                dict.set_item(key.bind(py), converted)?;
            }
            dict.into_any()
        }
    };
    Ok(converted)
}
