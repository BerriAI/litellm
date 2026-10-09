use pyo3::{prelude::*, types::PyDict};

pub fn present<'py>(
    arguments: &Bound<'py, PyDict>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(arguments.get_item(name)?.filter(|value| !value.is_none()))
}
