use std::path::PathBuf;

use bytes::Bytes;
use litellm_core::ocr::{OcrDocumentInput, OcrFileContent};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    pybacked::PyBackedBytes,
    types::{PyBytes, PyString},
};

#[derive(Debug)]
pub(super) struct PythonFileReader {
    reader: Py<PyAny>,
    name: Option<String>,
}

impl PythonFileReader {
    pub(super) fn read(&self, py: Python<'_>) -> PyResult<OcrFileContent> {
        let value = self.reader.bind(py).call0()?;
        let bytes = if value.is_instance_of::<PyString>() {
            Bytes::from(value.extract::<String>()?)
        } else if value.is_instance_of::<PyBytes>() {
            extract_bytes(&value)?
        } else {
            return Err(PyTypeError::new_err(format!(
                "OCR file read must return bytes or str, got {}",
                value.get_type(),
            )));
        };
        Ok(OcrFileContent {
            bytes,
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
    use pyo3::types::PyDict;

    use super::*;

    fn eval<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        locals
    }

    #[test]
    fn extraction_validates_required_file_and_optional_mime_type() {
        Python::initialize();
        Python::attach(|py| {
            for expression in [c"{}", c"{'file': None}"] {
                let document = py.eval(expression, None, None).unwrap();
                let error = document.extract::<FileDocumentInput>().err().unwrap();
                assert!(error.is_instance_of::<PyValueError>(py));
                assert!(error.to_string().contains("must include a 'file' field"));
            }
            for expression in [
                c"{'file': b'abc', 'mime_type': None}",
                c"{'file': b'abc', 'mime_type': 7}",
            ] {
                let document = py.eval(expression, None, None).unwrap();
                let error = document.extract::<FileDocumentInput>().err().unwrap();
                assert!(error.is_instance_of::<PyTypeError>(py));
            }
            let error = py
                .eval(c"{'file': 'scan.pdf'}", None, None)
                .unwrap()
                .extract::<FileDocumentInput>()
                .err()
                .unwrap();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("bare str"));
            let document = py
                .eval(c"{'file': b'abc', 'mime_type': 'image/png'}", None, None)
                .unwrap();
            let input: FileDocumentInput = document.extract().unwrap();
            assert!(input.reader.is_none());
            assert_eq!(
                input.input,
                OcrDocumentInput::Bytes {
                    bytes: b"abc".as_slice().into(),
                    file_name: None,
                    mime_type: Some("image/png".into()),
                }
            );
        });
    }

    #[test]
    fn paths_and_readers_are_projected_without_io() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"from pathlib import Path
class Reader:
    name = 'scan.png'
    def __init__(self):
        self.reads = 0
    def read(self):
        self.reads += 1
        return b'abc'
reader = Reader()
document = {'file': reader, 'mime_type': 7}
reader_document = {'file': reader}
path_document = {'file': Path('/nonexistent/ocr-projection-test.pdf'), 'mime_type': 'image/png'}",
            );
            let document = locals.get_item("document").unwrap().unwrap();
            let error = document.extract::<FileDocumentInput>().err().unwrap();
            assert!(error.is_instance_of::<PyTypeError>(py));

            let document = locals.get_item("reader_document").unwrap().unwrap();
            let input: FileDocumentInput = document.extract().unwrap();
            assert_eq!(
                input.input,
                OcrDocumentInput::HostReader { mime_type: None }
            );
            let reads = || {
                locals
                    .get_item("reader")
                    .unwrap()
                    .unwrap()
                    .getattr("reads")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap()
            };
            assert_eq!(reads(), 0);
            let content = input.reader.unwrap().read(py).unwrap();
            assert_eq!(reads(), 1);
            assert_eq!(
                content,
                OcrFileContent {
                    bytes: b"abc".as_slice().into(),
                    file_name: Some("scan.png".into()),
                }
            );

            let document = locals.get_item("path_document").unwrap().unwrap();
            let input: FileDocumentInput = document.extract().unwrap();
            assert!(input.reader.is_none());
            assert_eq!(
                input.input,
                OcrDocumentInput::Path {
                    path: PathBuf::from("/nonexistent/ocr-projection-test.pdf"),
                    mime_type: Some("image/png".into()),
                }
            );
        });
    }

    #[test]
    fn reader_results_are_normalized_and_exceptions_keep_their_identity() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"failure = KeyError('reader failed')
class Raising:
    def read(self):
        raise failure
class Text:
    def read(self):
        return 'héllo'
class Wrong:
    def read(self):
        return 7
raising = {'file': Raising()}
text = {'file': Text()}
wrong = {'file': Wrong()}",
            );
            let reader = |name: &str| {
                locals
                    .get_item(name)
                    .unwrap()
                    .unwrap()
                    .extract::<FileDocumentInput>()
                    .unwrap()
                    .reader
                    .unwrap()
            };
            let error = reader("raising").read(py).unwrap_err();
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
            assert_eq!(
                reader("text").read(py).unwrap().bytes.as_ref(),
                "héllo".as_bytes()
            );
            let error = reader("wrong").read(py).unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
            assert!(error.to_string().contains("bytes or str"));
        });
    }

    #[rstest::rstest]
    #[case::read("read")]
    #[case::name("name")]
    fn reader_attribute_failures_keep_their_identity(#[case] attribute: &str) {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"failure = LookupError('file property failed')
class File:
    def __getattribute__(self, name):
        if name == attribute:
            raise failure
        return super().__getattribute__(name)
    name = 'scan.pdf'
    def read(self):
        return b'abc'
document = {'file': File()}",
            );
            locals.set_item("attribute", attribute).unwrap();
            let error = locals
                .get_item("document")
                .unwrap()
                .unwrap()
                .extract::<FileDocumentInput>()
                .err()
                .unwrap();
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
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
