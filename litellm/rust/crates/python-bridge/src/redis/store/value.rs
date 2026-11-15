use litellm_python_compat::{BigInt, MAX_DEPTH, Value};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{
        PyBool, PyBytes, PyComplex, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PySet, PyString,
        PyTuple,
    },
};

/// A plain Python value (what `json.loads` or `ast.literal_eval` returns) across the bridge.
/// Anything else, a live object included, is refused with `TypeError`.
pub(crate) struct PyValue(pub(crate) Value);

impl<'a, 'py> FromPyObject<'a, 'py> for PyValue {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        to_value(&value, 0).map(Self)
    }
}

impl<'py> IntoPyObject<'py> for PyValue {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        to_python(py, self.0)
    }
}

fn to_value(value: &Bound<'_, PyAny>, depth: usize) -> PyResult<Value> {
    if depth > MAX_DEPTH {
        return Err(PyValueError::new_err("value is nested too deeply"));
    }
    let nested = |items: Bound<'_, PyAny>| -> PyResult<Vec<Value>> {
        items
            .try_iter()?
            .map(|item| to_value(&item?, depth + 1))
            .collect()
    };
    if value.is_none() {
        return Ok(Value::None);
    }
    if let Ok(flag) = value.cast::<PyBool>() {
        return Ok(Value::Bool(flag.is_true()));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(Value::Int(match value.extract::<i64>() {
            Ok(small) => small.into(),
            Err(_) => value
                .str()?
                .to_str()?
                .parse::<BigInt>()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        }));
    }
    if let Ok(number) = value.cast::<PyFloat>() {
        return Ok(Value::Float(number.value()));
    }
    if let Ok(text) = value.cast::<PyString>() {
        return Ok(Value::Str(text.to_str()?.to_owned()));
    }
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(Value::Bytes(bytes.as_bytes().to_vec()));
    }
    if let Ok(complex) = value.cast::<PyComplex>() {
        return Ok(Value::Complex {
            re: complex.real(),
            im: complex.imag(),
        });
    }
    if let Ok(entries) = value.cast::<PyDict>() {
        return entries
            .iter()
            .map(|(key, item)| Ok((to_value(&key, depth + 1)?, to_value(&item, depth + 1)?)))
            .collect::<PyResult<_>>()
            .map(Value::Dict);
    }
    if value.is_instance_of::<PyList>() {
        return nested(value.clone()).map(Value::List);
    }
    if value.is_instance_of::<PyTuple>() {
        return nested(value.clone()).map(Value::Tuple);
    }
    if value.is_instance_of::<PySet>() || value.is_instance_of::<PyFrozenSet>() {
        return nested(value.clone()).map(Value::Set);
    }
    Err(PyTypeError::new_err(format!(
        "{} is not a plain data value",
        value.get_type().name()?
    )))
}

fn to_python(py: Python<'_>, value: Value) -> PyResult<Bound<'_, PyAny>> {
    let all = |items: Vec<Value>| -> PyResult<Vec<Bound<'_, PyAny>>> {
        items.into_iter().map(|item| to_python(py, item)).collect()
    };
    Ok(match value {
        Value::None => py.None().into_bound(py),
        Value::Bool(flag) => PyBool::new(py, flag).to_owned().into_any(),
        Value::Int(number) => match i64::try_from(&number) {
            Ok(small) => small.into_pyobject(py)?.into_any(),
            Err(_) => py
                .get_type::<PyInt>()
                .call1((number.to_string(),))?
                .into_any(),
        },
        Value::Float(number) => number.into_pyobject(py)?.into_any(),
        Value::Complex { re, im } => PyComplex::from_doubles(py, re, im).into_any(),
        Value::Str(text) => text.into_pyobject(py)?.into_any(),
        Value::Bytes(bytes) => PyBytes::new(py, &bytes).into_any(),
        Value::Tuple(items) => PyTuple::new(py, all(items)?)?.into_any(),
        Value::List(items) => PyList::new(py, all(items)?)?.into_any(),
        Value::Set(items) => PySet::new(py, all(items)?)?.into_any(),
        Value::Dict(entries) => {
            let dict = PyDict::new(py);
            for (key, item) in entries {
                dict.set_item(to_python(py, key)?, to_python(py, item)?)?;
            }
            dict.into_any()
        }
    })
}
