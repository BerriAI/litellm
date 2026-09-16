use pyo3::prelude::*;
use pyo3::types::PyCFunction;

use pyo3::exceptions::PyRuntimeError;

mod audio_transcription;
mod chat_completions;
mod messages;
mod ocr;

#[cfg(test)]
mod tests;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ocr::register(module)?;
    audio_transcription::register(module)?;
    messages::register(module)?;
    chat_completions::register(module)?;
    Ok(())
}

pub(super) fn add_function(
    module: &Bound<'_, PyModule>,
    function: Bound<'_, PyCFunction>,
) -> PyResult<()> {
    let name: String = function.getattr("__name__")?.extract()?;
    if module.hasattr(&name)? {
        return Err(PyRuntimeError::new_err(format!(
            "duplicate native route: {name}"
        )));
    }
    module.add_function(function)
}
