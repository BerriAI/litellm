use litellm_core::ocr::{credential_default_fields, credential_index};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub(super) fn prepare<'py>(
    py: Python<'py>,
    kwargs: &Bound<'py, PyDict>,
    logger: &super::PythonLogger,
) -> PyResult<Bound<'py, PyDict>> {
    let arguments = kwargs.copy()?;
    arguments.set_item("litellm_logging_obj", logger.object(py))?;
    let litellm = py.import("litellm")?;
    inherit_credentials(py, &litellm, &arguments)?;
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("check_limits")?
        .call1((&arguments,))?;
    Ok(arguments)
}

fn inherit_credentials(
    py: Python<'_>,
    litellm: &Bound<'_, PyModule>,
    arguments: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let Some(requested) = arguments
        .get_item("litellm_credential_name")?
        .filter(|value| !value.is_none())
    else {
        return Ok(());
    };
    if !requested.is_truthy()? {
        return Ok(());
    }
    let requested: String = requested.extract()?;
    let credentials = litellm.getattr("credential_list")?.cast_into::<PyList>()?;
    let names = credentials
        .iter()
        .map(|credential| credential.getattr("credential_name")?.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?;
    let Some(index) = credential_index(&requested, &names) else {
        py.import("litellm._logging")?.getattr("verbose_logger")?.call_method1(
            "warning",
            ("litellm_credential_name=%s matched none of the %d loaded credentials; the request runs without it", requested, names.len()),
        )?;
        return Ok(());
    };
    let values = credentials
        .get_item(index)?
        .getattr("credential_values")?
        .cast_into::<PyDict>()?;
    let supplied: Vec<String> = arguments.keys().extract()?;
    let fields: Vec<String> = values.keys().extract()?;
    for name in credential_default_fields(&supplied, &fields) {
        if let Some(value) = values.get_item(name)? {
            arguments.set_item(name, value)?;
        }
    }
    Ok(())
}
