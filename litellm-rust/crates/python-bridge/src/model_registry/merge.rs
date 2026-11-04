use litellm_model_catalog::{FieldValue, Fields, Number, Shape};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};

/// A Python value under the registration merge rule. Values the rule does not rewrite keep their identity
struct PyField(Py<PyAny>);

impl Clone for PyField {
    fn clone(&self) -> Self {
        Python::attach(|py| Self(self.0.clone_ref(py)))
    }
}

impl FieldValue for PyField {
    fn shape(&self) -> Shape {
        Python::attach(|py| {
            let value = self.0.bind(py);
            if value.is_none() {
                return Shape::None;
            }
            if let Ok(text) = value.cast::<PyString>() {
                return Shape::Str(text.to_string_lossy().into_owned());
            }
            if value.is_instance_of::<PyDict>() {
                return Shape::Map;
            }
            Shape::Other
        })
    }

    fn none() -> Self {
        Python::attach(|py| Self(py.None()))
    }

    fn number(number: Number) -> Option<Self> {
        Python::attach(|py| {
            let value = match number {
                Number::Float(value) => PyFloat::new(py, value).into_any(),
                Number::Int(value) => py.get_type::<PyInt>().call1((value.to_string(),)).ok()?,
            };
            Some(Self(value.unbind()))
        })
    }

    fn merged_over(&self, base: Option<&Self>) -> Self {
        Python::attach(|py| {
            let merged = PyDict::new(py);
            let base = base
                .map(|base| base.0.bind(py))
                .filter(|base| base.is_instance_of::<PyDict>());
            let copied = base.map_or(Ok(()), |base| {
                merged.update(base.cast::<PyDict>().expect("a dict").as_mapping())
            });
            copied
                .and_then(|()| {
                    merged.update(
                        self.0
                            .bind(py)
                            .cast::<PyDict>()
                            .expect("a dict")
                            .as_mapping(),
                    )
                })
                .expect("dict.update of two dicts");
            Self(merged.into_any().unbind())
        })
    }

    fn rate_tiers(&self) -> Option<Vec<Fields<Self>>> {
        Python::attach(|py| {
            let tiers = self.0.bind(py);
            if !(tiers.is_instance_of::<PyList>() || tiers.is_instance_of::<PyTuple>()) {
                return None;
            }
            tiers
                .try_iter()
                .ok()?
                .map(|tier| tier.ok().and_then(|tier| rate_table(&tier)))
                .collect()
        })
    }

    fn rate_table(&self) -> Option<Fields<Self>> {
        Python::attach(|py| rate_table(self.0.bind(py)))
    }

    fn tier_list(tiers: Vec<Fields<Self>>) -> Self {
        Python::attach(|py| {
            let tiers: Vec<Bound<'_, PyDict>> =
                tiers.into_iter().map(|tier| to_dict(py, tier)).collect();
            Self(
                PyList::new(py, tiers)
                    .expect("a list of dicts")
                    .into_any()
                    .unbind(),
            )
        })
    }

    fn table(fields: Fields<Self>) -> Self {
        Python::attach(|py| Self(to_dict(py, fields).into_any().unbind()))
    }
}

/// The value as pydantic's `Mapping[str, object]` reads it, `None` where it raises
fn rate_table(value: &Bound<'_, PyAny>) -> Option<Fields<PyField>> {
    if !value.is_instance(mapping_type(value.py()).ok()?).ok()? {
        return None;
    }
    let items = value.call_method0("items").ok()?;
    items
        .try_iter()
        .ok()?
        .map(|item| {
            let (name, rate): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item.ok()?.extract().ok()?;
            Some((
                name.cast::<PyString>().ok()?.to_str().ok()?.to_owned(),
                PyField(rate.unbind()),
            ))
        })
        .collect()
}

fn mapping_type(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static MAPPING: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    MAPPING.import(py, "collections.abc", "Mapping")
}

fn to_dict(py: Python<'_>, fields: Fields<PyField>) -> Bound<'_, PyDict> {
    let dict = PyDict::new(py);
    for (name, value) in fields {
        dict.set_item(name, value.0).expect("str keys");
    }
    dict
}

/// A row's fields in dict order. A field name that is not a `str` is a `TypeError`
fn fields(row: &Bound<'_, PyAny>) -> PyResult<Fields<PyField>> {
    row.call_method0("items")?
        .try_iter()?
        .map(|item| {
            let (name, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
            let Ok(text) = name.cast::<PyString>() else {
                let type_name = name.get_type().name()?;
                return Err(PyTypeError::new_err(format!(
                    "field names must be str, got {type_name}"
                )));
            };
            let name = text.to_str()?.to_owned();
            Ok((name, PyField(value.unbind())))
        })
        .collect()
}

/// Main's `update_dictionary`, as a new dict whose keys are `existing`'s then `new`'s
#[pyfunction]
pub(crate) fn update_dictionary<'py>(
    existing: &Bound<'py, PyAny>,
    new: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let merged = litellm_model_catalog::update_dictionary(fields(existing)?, &fields(new)?);
    Ok(to_dict(existing.py(), merged))
}

/// The number a stringified registration value spells, as main's `_convert_stringified_numbers`
#[pyfunction]
pub(crate) fn stringified_number(py: Python<'_>, text: &str) -> Option<Py<PyAny>> {
    litellm_model_catalog::stringified_number(text)
        .and_then(PyField::number)
        .map(|value| value.0.clone_ref(py))
}

#[pyfunction]
pub(crate) fn registration_row<'py>(
    base: &Bound<'py, PyAny>,
    registration: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let row = litellm_model_catalog::registration_row(&fields(base)?, &fields(registration)?);
    Ok(to_dict(base.py(), row))
}

#[pyfunction]
pub(crate) fn without_overridden_price_variants<'py>(
    entry: &Bound<'py, PyAny>,
    registration: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let row = litellm_model_catalog::without_overridden_price_variants(
        &fields(entry)?,
        &fields(registration)?,
    );
    Ok(to_dict(entry.py(), row))
}

/// `raw_entry` is the `litellm.model_cost` row the lookup resolved to, whose fields decide which prices are real
#[pyfunction]
pub(crate) fn without_synthesized_costs<'py>(
    model_info: &Bound<'py, PyAny>,
    raw_entry: &Bound<'py, PyAny>,
    registration: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let row = litellm_model_catalog::without_synthesized_costs(
        &fields(model_info)?,
        |name| raw_entry.contains(name).unwrap_or(false),
        &fields(registration)?,
    );
    Ok(to_dict(model_info.py(), row))
}

#[pyfunction]
pub(crate) fn with_pricing_overrides<'py>(
    model_info: &Bound<'py, PyAny>,
    overrides: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let row =
        litellm_model_catalog::with_pricing_overrides(&fields(model_info)?, &fields(overrides)?);
    Ok(to_dict(model_info.py(), row))
}
