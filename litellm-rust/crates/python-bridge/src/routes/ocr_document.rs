use std::io::Read;
use std::path::PathBuf;

use pyo3::exceptions::{PyFileNotFoundError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBytes, PyDict, PyString};

use litellm_core::constants::OCR_INLINE_MAX_BYTES;
use litellm_core::ocr::{OcrDocument, encode_file_document, mime_type_for_name, upload_mime_type};
use litellm_python_interop::to_py_preserving_errors;

enum FileBytes {
    Python(PyBackedBytes),
    Native(Vec<u8>),
}

impl AsRef<[u8]> for FileBytes {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Python(bytes) => bytes,
            Self::Native(bytes) => bytes,
        }
    }
}

fn read_file_input(
    py: Python<'_>,
    file: &Bound<'_, PyAny>,
) -> PyResult<(FileBytes, Option<String>)> {
    if file.is_instance_of::<PyString>() {
        return Err(PyValueError::new_err(
            "OCR file input does not accept bare str values. Pass bytes, a pathlib.Path, or a file-like object.",
        ));
    }
    if file.is_instance(&py.import("os")?.getattr("PathLike")?)? {
        let path: PathBuf = file.extract()?;
        let name = path
            .file_name()
            .map(|value| value.to_string_lossy().into_owned());
        let bytes = py
            .detach(|| {
                let mut bytes = Vec::new();
                std::fs::File::open(&path)?
                    .take(OCR_INLINE_MAX_BYTES as u64 + 1)
                    .read_to_end(&mut bytes)?;
                Ok::<_, std::io::Error>(bytes)
            })
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    PyFileNotFoundError::new_err(format!("File not found: {}", path.display()))
                } else {
                    error.into()
                }
            })?;
        return Ok((FileBytes::Native(bytes), name));
    }
    if file.is_instance_of::<PyBytes>() {
        return Ok((FileBytes::Python(file.extract()?), None));
    }
    let reader = file
        .getattr_opt("read")?
        .filter(|value| value.is_callable());
    let Some(reader) = reader else {
        return Err(PyValueError::new_err(format!(
            "Unsupported file input type: {}. Expected pathlib.Path, bytes, or a file-like object.",
            file.get_type(),
        )));
    };
    let name = file
        .getattr_opt("name")?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<String>())
        .transpose()?;
    let value = reader.call0()?;
    let bytes = if value.is_instance_of::<PyString>() {
        FileBytes::Native(value.extract::<String>()?.into_bytes())
    } else if value.is_instance_of::<PyBytes>() {
        FileBytes::Python(value.extract()?)
    } else {
        return Err(PyTypeError::new_err(format!(
            "OCR file read must return bytes or str, got {}",
            value.get_type(),
        )));
    };
    Ok((bytes, name))
}

pub(super) fn file_document(py: Python<'_>, document: &Bound<'_, PyAny>) -> PyResult<OcrDocument> {
    let file = document.get_item("file").map_err(|error| {
        if error.is_instance_of::<pyo3::exceptions::PyKeyError>(py) {
            PyValueError::new_err("document with type='file' must include a 'file' field containing a pathlib.Path, file-like object, or bytes")
        } else {
            error
        }
    })?;
    if file.is_none() {
        return Err(PyValueError::new_err(
            "document with type='file' must include a 'file' field containing a pathlib.Path, file-like object, or bytes",
        ));
    }
    let (bytes, name) = read_file_input(py, &file)?;
    let mime = document
        .cast::<PyDict>()?
        .get_item("mime_type")?
        .map(|value| value.extract::<String>())
        .transpose()?;
    py.detach(|| encode_file_document(bytes.as_ref(), name.as_deref(), mime.as_deref()))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn _ocr_file_document(py: Python<'_>, document: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    to_py_preserving_errors(py, &file_document(py, &document)?)
}

#[pyfunction]
fn _ocr_mime_type(file_name: &str) -> String {
    mime_type_for_name(file_name).into()
}

#[pyfunction]
#[pyo3(signature = (file_content, file_name=None, content_type=None))]
fn _ocr_upload_document(
    py: Python<'_>,
    file_content: &Bound<'_, PyBytes>,
    file_name: Option<&str>,
    content_type: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let bytes: PyBackedBytes = file_content.extract()?;
    let document = py
        .detach(|| {
            encode_file_document(
                &bytes,
                None,
                Some(upload_mime_type(file_name, content_type)),
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    to_py_preserving_errors(py, &document)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("_OCR_MAX_FILE_BYTES", OCR_INLINE_MAX_BYTES)?;
    module.add_function(wrap_pyfunction!(_ocr_upload_document, module)?)?;
    module.add_function(wrap_pyfunction!(_ocr_file_document, module)?)?;
    module.add_function(wrap_pyfunction!(_ocr_mime_type, module)?)
}
