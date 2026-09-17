use std::path::PathBuf;

use bytes::Bytes;
use pyo3::exceptions::PyValueError;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBytes, PyString};

use litellm_core::ocr::{OcrDocumentInput, OcrFileContent};

pub(super) struct PythonFileReader {
    reader: Py<PyAny>,
    name: Option<String>,
}

impl PythonFileReader {
    pub(super) fn read(&self, py: Python<'_>) -> PyResult<OcrFileContent> {
        let value = py
            .import("litellm.rust_bridge.ocr")?
            .getattr("read_document")?
            .call1((self.reader.bind(py),))?;
        Ok(OcrFileContent {
            bytes: extract_bytes(&value)?,
            file_name: self.name.clone(),
        })
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.reader)
    }
}

fn extract_bytes(value: &Bound<'_, PyAny>) -> PyResult<Bytes> {
    if value.is_exact_instance_of::<PyBytes>() {
        return Ok(Bytes::from_owner(value.extract::<PyBackedBytes>()?));
    }
    // Bytes subclasses may retain GC edges that a native Bytes owner cannot traverse.
    Ok(Bytes::copy_from_slice(
        value.extract::<PyBackedBytes>()?.as_ref(),
    ))
}

pub(super) struct FileDocumentInput {
    pub input: OcrDocumentInput,
    pub reader: Option<PythonFileReader>,
}

impl FromPyObject<'_, '_> for FileDocumentInput {
    type Error = PyErr;

    fn extract(document: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let py = document.py();
        let mime_type = match document.get_item("mime_type") {
            Ok(value) => Some(value.extract::<String>()?),
            Err(error) if error.is_instance_of::<pyo3::exceptions::PyKeyError>(py) => None,
            Err(error) => return Err(error),
        };
        let missing = || {
            PyValueError::new_err(
                "document with type='file' must include a 'file' field containing a pathlib.Path, file-like object, or bytes",
            )
        };
        let file = document.get_item("file").map_err(|error| {
            if error.is_instance_of::<pyo3::exceptions::PyKeyError>(py) {
                missing()
            } else {
                error
            }
        })?;
        if file.is_none() {
            return Err(missing());
        }
        if file.is_instance_of::<PyString>() {
            return Err(PyValueError::new_err(
                "OCR file input does not accept bare str values. Pass bytes, a pathlib.Path, or a file-like object.",
            ));
        }
        if file.is_instance(&py.import("os")?.getattr("PathLike")?)? {
            return Ok(Self {
                input: OcrDocumentInput::Path {
                    path: file.extract::<PathBuf>()?,
                    mime_type,
                },
                reader: None,
            });
        }
        if file.is_instance_of::<PyBytes>() {
            return Ok(Self {
                input: OcrDocumentInput::Bytes {
                    bytes: extract_bytes(&file)?,
                    file_name: None,
                    mime_type,
                },
                reader: None,
            });
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
        Ok(Self {
            input: OcrDocumentInput::HostReader { mime_type },
            reader: Some(PythonFileReader {
                reader: reader.unbind(),
                name,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyTypeError;
    use pyo3::types::PyDict;

    #[test]
    fn projection_validates_fields_without_consuming_readers_or_opening_paths() {
        Python::initialize();
        Python::attach(|py| {
            for expression in [c"{}", c"{'file': None}"] {
                let document = py.eval(expression, None, None).unwrap();
                let error = document.extract::<FileDocumentInput>().err().unwrap();
                assert!(error.is_instance_of::<PyValueError>(py));
            }
            for expression in [
                c"{'file': b'abc', 'mime_type': None}",
                c"{'file': b'abc', 'mime_type': 7}",
            ] {
                let error = py
                    .eval(expression, None, None)
                    .unwrap()
                    .extract::<FileDocumentInput>()
                    .err()
                    .unwrap();
                assert!(error.is_instance_of::<PyTypeError>(py));
            }
            let locals = PyDict::new(py);
            py.run(
                c"from pathlib import Path
failure = KeyError('reader failed')
class Reader:
    def __init__(self):
        self.reads = 0
    def read(self):
        self.reads += 1
        raise failure
reader = Reader()
document = {'file': reader}
path_document = {'file': Path('/nonexistent/ocr-projection-test.pdf')}
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let document = locals.get_item("document").unwrap().unwrap();
            let input: FileDocumentInput = document.extract().unwrap();
            assert_eq!(
                locals
                    .get_item("reader")
                    .unwrap()
                    .unwrap()
                    .getattr("reads")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                0
            );
            assert!(input.reader.is_some());
            let path: FileDocumentInput = locals
                .get_item("path_document")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!(matches!(path.input, OcrDocumentInput::Path { .. }));
        });
    }

    #[test]
    fn exact_python_bytes_transfer_without_copying_and_outlive_the_input() {
        Python::initialize();
        let (bytes, pointer) = Python::attach(|py| {
            let value = PyBytes::new(py, b"document bytes");
            let pointer = value.as_bytes().as_ptr() as usize;
            (extract_bytes(value.as_any()).unwrap(), pointer)
        });
        assert_eq!(bytes.as_ptr() as usize, pointer);
        assert_eq!(bytes.as_ref(), b"document bytes");
    }
}
