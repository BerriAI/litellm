use pyo3::{
    IntoPyObjectExt,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString},
};
use serde_yaml_ng::{
    Mapping, Number, Value,
    value::{Tag, TaggedValue},
};

use super::Error;

const OPAQUE_TAG: &str = "python";
const MAX_DEPTH: usize = litellm_python_compat::MAX_DEPTH;

/// Python objects a config carries that YAML cannot hold, such as a `datetime.date` or an int past 64 bits.
/// Each one crosses to Rust as a tagged index into this list and comes back as the same object.
#[derive(Default)]
pub(super) struct Opaque(Vec<Py<PyAny>>);

impl Opaque {
    fn keep(&mut self, value: &Bound<'_, PyAny>) -> Value {
        self.0.push(value.clone().unbind());
        Value::Tagged(Box::new(TaggedValue {
            tag: Tag::new(OPAQUE_TAG),
            value: Value::from(self.0.len() - 1),
        }))
    }

    fn restore<'py>(&self, py: Python<'py>, tagged: &TaggedValue) -> PyResult<Bound<'py, PyAny>> {
        tagged
            .value
            .as_u64()
            .and_then(|index| usize::try_from(index).ok())
            .and_then(|index| self.0.get(index))
            .map(|kept| kept.bind(py).clone())
            .ok_or_else(|| Error::LostValue.into())
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.0.iter().try_for_each(|kept| visit.call(kept))
    }
}

pub(super) fn from_python(value: &Bound<'_, PyAny>, opaque: &mut Opaque) -> PyResult<Value> {
    nested_from_python(value, opaque, 0)
}

fn nested_from_python(
    value: &Bound<'_, PyAny>,
    opaque: &mut Opaque,
    depth: usize,
) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if value.is_exact_instance_of::<PyBool>() {
        return Ok(Value::Bool(value.extract()?));
    }
    if value.is_exact_instance_of::<PyInt>() {
        return Ok(match (value.extract::<i64>(), value.extract::<u64>()) {
            (Ok(int), _) => Value::from(int),
            (_, Ok(int)) => Value::from(int),
            _ => opaque.keep(value),
        });
    }
    if value.is_exact_instance_of::<PyFloat>() {
        return Ok(Value::from(value.extract::<f64>()?));
    }
    if value.is_exact_instance_of::<PyString>() {
        return Ok(match value.extract::<String>() {
            Ok(text) => Value::String(text),
            Err(_) => opaque.keep(value),
        });
    }
    if depth == MAX_DEPTH {
        return Ok(opaque.keep(value));
    }
    if let Ok(list) = value.cast::<PyList>() {
        return list
            .iter()
            .map(|item| nested_from_python(&item, opaque, depth + 1))
            .collect::<PyResult<_>>()
            .map(Value::Sequence);
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        return dict
            .iter()
            .map(|(key, item)| {
                Ok((
                    nested_from_python(&key, opaque, depth + 1)?,
                    nested_from_python(&item, opaque, depth + 1)?,
                ))
            })
            .collect::<PyResult<Mapping>>()
            .map(Value::Mapping);
    }
    Ok(opaque.keep(value))
}

pub(super) fn to_python<'py>(
    py: Python<'py>,
    value: &Value,
    opaque: &Opaque,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Value::Null => Ok(py.None().into_bound(py)),
        Value::Bool(flag) => (*flag).into_bound_py_any(py),
        Value::Number(number) => number_to_python(py, number),
        Value::String(text) => text.as_str().into_bound_py_any(py),
        Value::Sequence(items) => {
            let list = PyList::empty(py);
            items
                .iter()
                .try_for_each(|item| list.append(to_python(py, item, opaque)?))?;
            Ok(list.into_any())
        }
        Value::Mapping(mapping) => {
            let dict = PyDict::new(py);
            mapping.iter().try_for_each(|(key, item)| {
                dict.set_item(to_python(py, key, opaque)?, to_python(py, item, opaque)?)
            })?;
            Ok(dict.into_any())
        }
        Value::Tagged(tagged) => opaque.restore(py, tagged),
    }
}

fn number_to_python<'py>(py: Python<'py>, number: &Number) -> PyResult<Bound<'py, PyAny>> {
    match (number.as_i64(), number.as_u64()) {
        (Some(int), _) => int.into_bound_py_any(py),
        (_, Some(int)) => int.into_bound_py_any(py),
        _ => number.as_f64().into_bound_py_any(py),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::scalars(
        c"{'a': 1, 'b': 1.0, 'c': True, 'd': None, 'e': 'text', 'f': -0.0, 'g': 2**63}"
    )]
    #[case::nested(
        c"{'model_list': [{'model_name': 'm', 'litellm_params': {'model': 'x', 'tags': []}}]}"
    )]
    #[case::non_text_keys(c"{1: 'int', 2.5: 'float', None: 'none', False: 'bool'}")]
    #[case::key_order(c"{'z': 1, 'a': 2, 'm': 3}")]
    #[case::values_yaml_cannot_hold(
        c"{'day': __import__('datetime').date(2024, 2, 1), 'big': 2**80, 'pair': (1, 2), 'raw': b'bytes'}"
    )]
    fn a_document_comes_back_to_python_as_it_went_in(#[case] expression: &CStr) {
        Python::initialize();
        Python::attach(|py| {
            let original = py.eval(expression, None, None).unwrap();
            let mut opaque = Opaque::default();

            let document = from_python(&original, &mut opaque).unwrap();
            let restored = to_python(py, &document, &opaque).unwrap();

            assert_eq!(
                restored.repr().unwrap().to_string(),
                original.repr().unwrap().to_string()
            );
        });
    }

    #[rstest]
    fn a_value_yaml_cannot_hold_comes_back_as_the_same_object() {
        Python::initialize();
        Python::attach(|py| {
            let original = py
                .eval(
                    c"{'day': __import__('datetime').date(2024, 2, 1)}",
                    None,
                    None,
                )
                .unwrap();
            let mut opaque = Opaque::default();

            let document = from_python(&original, &mut opaque).unwrap();
            let restored = to_python(py, &document, &opaque).unwrap();

            assert!(
                restored
                    .get_item("day")
                    .unwrap()
                    .is(&original.get_item("day").unwrap())
            );
        });
    }

    #[rstest]
    fn a_list_that_contains_itself_stops_at_the_depth_limit() {
        Python::initialize();
        Python::attach(|py| {
            let cyclic = py
                .eval(
                    c"(lambda items: items.append(items) or items)([])",
                    None,
                    None,
                )
                .unwrap();
            let mut opaque = Opaque::default();

            let document = from_python(&cyclic, &mut opaque).unwrap();
            let restored = to_python(py, &document, &opaque).unwrap();

            assert_eq!(restored.len().unwrap(), 1);
        });
    }
}
