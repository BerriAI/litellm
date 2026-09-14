mod callbacks;
mod document;
mod errors;
mod lifecycle;
mod project;
mod value;

use pyo3::prelude::*;

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    value::register(module)?;
    document::register(module)?;
    lifecycle::register(module)
}

#[cfg(feature = "trace-parity")]
pub(super) fn register_trace(module: &Bound<'_, PyModule>) -> PyResult<()> {
    value::register_trace(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_preserves_existing_private_exports() {
        Python::initialize();
        Python::attach(|py| {
            for name in [
                "_ocr_lifecycle",
                "_ocr_upload_document",
                "_ocr_file_document",
                "_ocr_mime_type",
            ] {
                let module = PyModule::new(py, "ocr").unwrap();
                let original = pyo3::types::PyDict::new(py);
                module.add(name, &original).unwrap();
                let error = register(&module).unwrap_err();
                assert!(error.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
                assert_eq!(
                    error.value(py).to_string(),
                    format!("duplicate native route: {name}")
                );
                assert!(module.getattr(name).unwrap().is(&original));
            }
        });
    }
}
