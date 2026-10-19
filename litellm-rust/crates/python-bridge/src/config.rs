use std::path::Path;

use litellm_config::{
    Document,
    includes::{self, IncludePlan, IncludeValue},
};
use litellm_host_python::{release_gil, to_py};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};

struct PythonValue<'py>(Bound<'py, PyAny>);

impl IncludeValue for PythonValue<'_> {
    type Failure = PyErr;

    fn sequence(&self) -> Option<Vec<Self>> {
        self.0
            .cast::<PyList>()
            .ok()
            .map(|values| values.iter().map(Self).collect())
    }

    fn string(&self) -> Option<String> {
        self.0.extract().ok()
    }

    fn with_sequence(&self, values: Vec<Self>) -> PyResult<Self> {
        Ok(Self(
            PyList::new(self.0.py(), values.into_iter().map(|value| value.0))?.into_any(),
        ))
    }
}

fn entries<'py>(config: &Bound<'py, PyDict>) -> PyResult<Vec<(String, PythonValue<'py>)>> {
    config
        .iter()
        .map(|(key, value)| Ok((key.extract()?, PythonValue(value))))
        .collect()
}

fn paths(config: &Bound<'_, PyDict>) -> PyResult<Vec<String>> {
    let value = config.get_item("include")?.map(PythonValue);
    includes::include_entries(value.as_ref())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyclass(name = "_ConfigIncludes", module = "litellm.rust_bridge._native")]
pub(crate) struct ConfigIncludes {
    plan: IncludePlan,
}

#[pymethods]
impl ConfigIncludes {
    #[new]
    fn new(config: &Bound<'_, PyDict>, location: String) -> PyResult<Self> {
        Ok(Self {
            plan: IncludePlan::new(location, paths(config)?),
        })
    }

    fn next_entry(&mut self) -> Option<(String, String)> {
        self.plan.pop()
    }

    fn mark_loaded(&mut self, location: String) -> bool {
        self.plan.mark_loaded(location)
    }

    fn merge<'py>(
        &mut self,
        config: &Bound<'py, PyDict>,
        included: &Bound<'py, PyDict>,
        location: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let paths = paths(included)?;
        let merged = includes::merge(entries(config)?, entries(included)?)?;
        let output = PyDict::new(config.py());
        for (key, value) in merged {
            output.set_item(key, value.0)?;
        }
        self.plan.extend(location, paths);
        Ok(output)
    }
}

#[pyfunction]
pub(crate) fn _parse_config_yaml(py: Python<'_>, yaml: String) -> PyResult<(bool, Py<PyAny>)> {
    match release_gil(py, || Document::from_yaml(&yaml)) {
        Ok(document) if document.python_compatible() => Ok((true, to_py(py, document.value())?)),
        _ => Ok((false, py.None())),
    }
}

#[pyfunction]
pub(crate) fn _resolve_config_include_path(
    py: Python<'_>,
    entry: String,
    declared_in: String,
    root: String,
) -> PyResult<(String, Option<String>)> {
    let resolved = release_gil(py, || {
        includes::resolve_include_file_path(&entry, Path::new(&declared_in), Path::new(&root))
    })
    .map_err(|error| PyErr::new::<pyo3::exceptions::PyOSError, _>(error.to_string()))?;
    Ok((
        resolved.location.to_string_lossy().into_owned(),
        resolved.warning,
    ))
}
