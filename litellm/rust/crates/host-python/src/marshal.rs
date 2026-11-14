use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

use pyo3::exceptions::PyValueError;
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Number, Value};

/// Converts a `#[pyo3(from_py_with = ...)]` argument, reporting failures as `ValueError`
/// so a bad argument reads as a bad argument rather than as whatever the conversion hit.
pub fn from_py_argument<T>(value: &Bound<'_, PyAny>) -> PyResult<T>
where
    T: DeserializeOwned,
{
    pythonize::depythonize(value)
        .map_err(|error| PyValueError::new_err(litellm_error::Report::new(&error).to_string()))
}

pub fn from_py<T>(value: &Bound<'_, PyAny>) -> PyResult<T>
where
    T: DeserializeOwned,
{
    pythonize::depythonize(value).map_err(PyErr::from)
}

pub fn to_py<T>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>>
where
    T: Serialize + ?Sized,
{
    Pythonized(value).into_pyobject(py).map(Bound::unbind)
}

pub fn json_object_field(py: Python<'_>, document: &str, name: &str) -> PyResult<Py<PyAny>> {
    py.import("json")?
        .call_method1("loads", (document,))?
        .call_method1("get", (name,))
        .map(Bound::unbind)
}

pub fn json_loads(py: Python<'_>, document: &[u8]) -> PyResult<Py<PyAny>> {
    py.import("json")?
        .call_method1("loads", (PyBytes::new(py, document),))
        .map(Bound::unbind)
}

pub struct Pythonized<T>(pub T);

impl<'py, T> IntoPyObject<'py> for Pythonized<T>
where
    T: Serialize,
{
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        catch_unwind(AssertUnwindSafe(|| pythonize::pythonize(py, &self.0)))
            .map_err(panic_to_pyerr)?
            .map_err(PyErr::from)
    }
}

pub fn panic_to_pyerr(payload: Box<dyn Any + Send>) -> PyErr {
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("panic from Rust code");
    PanicException::new_err(message.to_string())
}

pub fn json_from_py(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(flag) = value.cast::<PyBool>() {
        return Ok(Value::Bool(flag.is_true()));
    }
    if let Ok(int) = value.cast::<PyInt>() {
        if let Ok(number) = int.extract::<i64>() {
            return Ok(Value::from(number));
        }
        if let Ok(number) = int.extract::<u64>() {
            return Ok(Value::from(number));
        }
        return raw_leaf(value);
    }
    if let Ok(float) = value.cast::<PyFloat>() {
        return match Number::from_f64(float.value()) {
            Some(number) => Ok(Value::Number(number)),
            None => raw_leaf(value),
        };
    }
    if let Ok(text) = value.cast::<PyString>() {
        return json_string(text);
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        if !dict
            .keys()
            .iter()
            .all(|key| key.is_instance_of::<PyString>())
        {
            return from_py(value);
        }
        return dict
            .iter()
            .map(|(key, item)| Ok((json_key(key.cast::<PyString>()?)?, json_from_py(&item)?)))
            .collect::<PyResult<Map<String, Value>>>()
            .map(Value::Object);
    }
    if let Ok(list) = value.cast::<PyList>() {
        return list
            .iter()
            .map(|item| json_from_py(&item))
            .collect::<PyResult<_>>()
            .map(Value::Array);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return tuple
            .iter()
            .map(|item| json_from_py(&item))
            .collect::<PyResult<_>>()
            .map(Value::Array);
    }
    from_py(value)
}

pub fn json_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(flag) => PyBool::new(py, *flag).to_owned().into_any(),
        Value::Number(number) => match (number.as_i64(), number.as_u64()) {
            (Some(int), _) => int.into_pyobject(py)?.into_any(),
            (None, Some(int)) => int.into_pyobject(py)?.into_any(),
            (None, None) => PyFloat::new(py, number.as_f64().unwrap_or(f64::NAN)).into_any(),
        },
        Value::String(text) => match litellm_host::raw_json::decode(text) {
            Some(raw) => py.import("json")?.call_method1("loads", (raw,))?,
            None => PyString::new(py, text).into_any(),
        },
        Value::Array(items) => PyList::new(
            py,
            items
                .iter()
                .map(|item| json_to_py(py, item))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any(),
        Value::Object(fields) => {
            let dict = PyDict::new(py);
            for (key, item) in fields {
                dict.set_item(
                    json_to_py(py, &Value::String(key.clone()))?,
                    json_to_py(py, item)?,
                )?;
            }
            dict.into_any()
        }
    })
}

fn json_key(text: &Bound<'_, PyString>) -> PyResult<String> {
    match json_string(text)? {
        Value::String(key) => Ok(key),
        _ => unreachable!("a Python str converts to a JSON string"),
    }
}

fn json_string(text: &Bound<'_, PyString>) -> PyResult<Value> {
    match text.to_str() {
        Ok(text) if !litellm_host::raw_json::is_marked(text) => Ok(Value::String(text.to_owned())),
        _ => raw_leaf(text.as_any()),
    }
}

fn raw_leaf(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let raw: String = value
        .py()
        .import("json")?
        .call_method1("dumps", (value,))?
        .extract()?;
    Ok(Value::String(litellm_host::raw_json::encode(&raw)))
}

#[cfg(test)]
mod tests {
    use serde::Serializer;

    use super::*;

    struct PanickingSerializer;

    impl Serialize for PanickingSerializer {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            panic!("serializer panicked")
        }
    }

    #[test]
    fn pythonized_converts_on_the_attached_thread() {
        crate::initialize_python();
        Python::attach(|py| {
            let value: Vec<i32> = Pythonized(vec![1, 2, 3])
                .into_pyobject(py)
                .and_then(|value| value.extract())
                .expect("value should convert");
            assert_eq!(value, vec![1, 2, 3]);
        });
    }

    #[test]
    fn pythonized_preserves_python_serialization_error_types() {
        crate::initialize_python();
        Python::attach(|py| {
            let value = std::collections::BTreeMap::from([(vec![1], "value")]);
            let direct = to_py(py, &value).unwrap_err();
            let wrapped = Pythonized(value).into_pyobject(py).unwrap_err();
            assert!(direct.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert!(wrapped.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
            assert_eq!(wrapped.to_string(), direct.to_string());
        });
    }

    #[rstest::rstest]
    #[case::wrapped(false)]
    #[case::direct(true)]
    fn output_conversion_maps_serializer_panics_to_a_base_exception(#[case] direct: bool) {
        crate::initialize_python();
        Python::attach(|py| {
            let error = if direct {
                to_py(py, &PanickingSerializer).unwrap_err()
            } else {
                Pythonized(PanickingSerializer)
                    .into_pyobject(py)
                    .unwrap_err()
            };
            assert!(error.is_instance_of::<PanicException>(py));
            assert_eq!(error.to_string(), "PanicException: serializer panicked");
        });
    }

    #[rstest::rstest]
    #[case::plain(
        c"{'a': [1, 2.5, None, True, 'x'], 'b': (1,)}",
        r#"{"a":[1,2.5,null,true,"x"],"b":[1]}"#
    )]
    #[case::nan(c"{'t': float('nan')}", r#"{"t":NaN}"#)]
    #[case::infinities(c"[float('inf'), -float('inf')]", "[Infinity,-Infinity]")]
    #[case::wide_integers(
        c"[2**64, -(2**70), 2**64 - 1, -(2**63)]",
        "[18446744073709551616,-1180591620717411303424,18446744073709551615,-9223372036854775808]"
    )]
    #[case::lone_surrogates(c"{'k\\udc00': 'a\\ud800b \\u00e9'}", r#"{"k\udc00":"a\ud800b é"}"#)]
    #[case::marked_text(c"['\\uf8fflitellm-raw-json:1']", r#"["litellm-raw-json:1"]"#)]
    fn json_values_keep_what_serde_json_cannot_hold(
        #[case] source: &std::ffi::CStr,
        #[case] wire: &str,
    ) {
        crate::initialize_python();
        Python::attach(|py| {
            let value = py.eval(source, None, None).unwrap();
            let converted = json_from_py(&value).unwrap();
            let json = py.import("json").unwrap();
            let loads = |text: &str| json.call_method1("loads", (text,)).unwrap();
            let wire_bytes = litellm_host::raw_json::to_vec(&converted).unwrap();
            assert!(
                loads(std::str::from_utf8(&wire_bytes).unwrap())
                    .eq(loads(wire))
                    .unwrap()
                    || wire.contains("NaN")
            );
            assert_eq!(
                json.call_method1("dumps", (loads(std::str::from_utf8(&wire_bytes).unwrap()),))
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                json.call_method1("dumps", (loads(wire),))
                    .unwrap()
                    .extract::<String>()
                    .unwrap()
            );
            let back = json_to_py(py, &converted).unwrap();
            assert_eq!(
                json.call_method1("dumps", (back,))
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                json.call_method1("dumps", (loads(wire),))
                    .unwrap()
                    .extract::<String>()
                    .unwrap()
            );
        });
    }

    #[test]
    fn values_json_cannot_hold_keep_their_old_conversion() {
        crate::initialize_python();
        Python::attach(|py| {
            let value = py.eval(c"{1: 'a'}", None, None).unwrap();
            assert_eq!(
                json_from_py(&value).unwrap_err().to_string(),
                from_py::<Value>(&value).unwrap_err().to_string()
            );
        });
    }

    #[test]
    fn depythonize_preserves_python_exception_identity_and_traceback() {
        crate::initialize_python();
        Python::attach(|py| {
            let locals = pyo3::types::PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
failure = LookupError('conversion failed')
cause = ValueError('cause')
class Broken:
    def __index__(self):
        raise failure from cause
value = Broken()
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let value = locals.get_item("value").unwrap().unwrap();
            let argument_error = from_py_argument::<i64>(&value).unwrap_err();
            assert!(argument_error.is_instance_of::<PyValueError>(py));
            assert!(
                !argument_error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
            let error = from_py::<i64>(&value).unwrap_err();
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
            assert!(
                error
                    .cause(py)
                    .unwrap()
                    .value(py)
                    .is(locals.get_item("cause").unwrap().unwrap())
            );
            assert!(error.traceback(py).is_some());
        });
    }
}
