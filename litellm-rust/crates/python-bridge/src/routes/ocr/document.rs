use std::path::PathBuf;

use litellm_host_python::{PythonFileReader, py_bytes};
use litellm_inference_ocr::types::OcrDocumentInput;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    sync::PyOnceLock,
    types::{PyBytes, PyString, PyType},
};

/// A `type='file'` document as projected: paths and bytes are typed inputs already; a
/// file-like object is a reader the projection consumes once every other field is read.
pub(super) enum FileDocumentInput {
    Ready(OcrDocumentInput),
    Deferred {
        reader: PythonFileReader,
        mime_type: Option<String>,
    },
}

impl FileDocumentInput {
    pub(super) fn resolve(self, py: Python<'_>) -> PyResult<OcrDocumentInput> {
        match self {
            Self::Ready(input) => Ok(input),
            Self::Deferred { reader, mime_type } => {
                let content = reader.read(py)?;
                Ok(OcrDocumentInput::Bytes {
                    bytes: content.bytes,
                    file_name: content.file_name,
                    mime_type,
                })
            }
        }
    }
}

fn optional_field<'py>(
    document: Borrowed<'_, 'py, PyAny>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if !document.contains(key)? {
        return Ok(None);
    }
    document.get_item(key).map(Some)
}

impl FromPyObject<'_, '_> for FileDocumentInput {
    type Error = PyErr;

    fn extract(document: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let py = document.py();
        let mime_type = optional_field(document, "mime_type")?
            .map(|value| value.extract::<String>())
            .transpose()?;
        let missing = || {
            PyValueError::new_err(
                "document with type='file' must include a 'file' field containing a pathlib.Path, file-like object, or bytes",
            )
        };
        let file = optional_field(document, "file")?.ok_or_else(missing)?;
        if file.is_none() {
            return Err(missing());
        }
        if file.is_instance_of::<PyString>() {
            return Err(PyValueError::new_err(
                "OCR file input does not accept bare str values. Pass bytes, a pathlib.Path, or a file-like object.",
            ));
        }
        static PATH_LIKE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        if file.is_instance(PATH_LIKE.import(py, "os", "PathLike")?)? {
            return Ok(Self::Ready(OcrDocumentInput::Path {
                path: file.extract::<PathBuf>()?,
                mime_type,
            }));
        }
        if file.is_instance_of::<PyBytes>() {
            return Ok(Self::Ready(OcrDocumentInput::Bytes {
                bytes: py_bytes(&file)?,
                file_name: None,
                mime_type,
            }));
        }
        match PythonFileReader::from_file_like(&file)? {
            Some(reader) => Ok(Self::Deferred { reader, mime_type }),
            None => Err(PyValueError::new_err(format!(
                "Unsupported file input type: {}. Expected pathlib.Path, bytes, or a file-like object.",
                file.get_type(),
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{exceptions::PyTypeError, types::PyDict};

    use super::*;

    fn eval<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        locals
    }

    fn ready(input: FileDocumentInput) -> OcrDocumentInput {
        match input {
            FileDocumentInput::Ready(input) => input,
            FileDocumentInput::Deferred { .. } => panic!("expected a ready document"),
        }
    }

    #[rstest::rstest]
    #[case::without_mime_type(c"{'file': b'abc'}", None)]
    #[case::with_mime_type(c"{'file': b'abc', 'mime_type': 'image/png'}", Some("image/png"))]
    fn a_dict_answering_missing_keys_reads_only_the_keys_it_holds(
        #[case] items: &std::ffi::CStr,
        #[case] mime_type: Option<&str>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"class Defaulting(dict):
    def __missing__(self, key):
        return Defaulting()
    def __getattr__(self, name):
        return self[name]",
            );
            let document = locals
                .get_item("Defaulting")
                .unwrap()
                .unwrap()
                .call1((py.eval(items, None, None).unwrap(),))
                .unwrap();
            assert_eq!(
                ready(document.extract().unwrap()),
                OcrDocumentInput::Bytes {
                    bytes: b"abc".as_slice().into(),
                    file_name: None,
                    mime_type: mime_type.map(str::to_owned),
                }
            );
            let error = locals
                .get_item("Defaulting")
                .unwrap()
                .unwrap()
                .call0()
                .unwrap()
                .extract::<FileDocumentInput>()
                .err()
                .unwrap();
            assert!(
                error.to_string().contains("must include a 'file' field"),
                "{error}"
            );
        });
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
            let error = py
                .eval(c"{'file': object()}", None, None)
                .unwrap()
                .extract::<FileDocumentInput>()
                .err()
                .unwrap();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("Unsupported file input type"));
            let document = py
                .eval(c"{'file': b'abc', 'mime_type': 'image/png'}", None, None)
                .unwrap();
            assert_eq!(
                ready(document.extract().unwrap()),
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
reader_document = {'file': reader, 'mime_type': 'application/pdf'}
path_document = {'file': Path('/nonexistent/ocr-projection-test.pdf'), 'mime_type': 'image/png'}",
            );
            let document = locals.get_item("document").unwrap().unwrap();
            let error = document.extract::<FileDocumentInput>().err().unwrap();
            assert!(error.is_instance_of::<PyTypeError>(py));

            let document = locals.get_item("reader_document").unwrap().unwrap();
            let input: FileDocumentInput = document.extract().unwrap();
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
            let resolved = input.resolve(py).unwrap();
            assert_eq!(reads(), 1);
            assert_eq!(
                resolved,
                OcrDocumentInput::Bytes {
                    bytes: b"abc".as_slice().into(),
                    file_name: Some("scan.png".into()),
                    mime_type: Some("application/pdf".into()),
                }
            );

            let document = locals.get_item("path_document").unwrap().unwrap();
            assert_eq!(
                ready(document.extract().unwrap()),
                OcrDocumentInput::Path {
                    path: PathBuf::from("/nonexistent/ocr-projection-test.pdf"),
                    mime_type: Some("image/png".into()),
                }
            );
        });
    }
}
