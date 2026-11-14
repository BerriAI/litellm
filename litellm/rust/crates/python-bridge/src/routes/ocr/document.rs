use std::path::PathBuf;

use litellm_host_python::{PythonFileReader, py_bytes};
use litellm_inference_ocr::types::OcrDocumentInput;
use pyo3::{
    exceptions::{PyKeyError, PyValueError},
    prelude::*,
    sync::PyOnceLock,
    types::{PyBytes, PyMapping, PyString, PyType},
};

/// A `type='file'` document the bridge cannot read, with the text Python's OCR path raises.
#[derive(Debug, thiserror::Error)]
enum DocumentError {
    #[error(
        "document with type='file' must include a 'file' field containing a pathlib.Path, file-like object, or bytes"
    )]
    MissingFile,
    #[error(
        "OCR file input does not accept bare str values. Pass bytes, a pathlib.Path, or a file-like object."
    )]
    BareString,
    #[error(
        "Unsupported file input type: {0}. Expected pathlib.Path, bytes, or a file-like object."
    )]
    UnsupportedType(String),
}

impl From<DocumentError> for PyErr {
    fn from(error: DocumentError) -> Self {
        PyValueError::new_err(litellm_error::Report::new(&error).to_string())
    }
}

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
    if document.cast::<PyMapping>().is_ok() && !document.contains(key)? {
        return Ok(None);
    }
    match document.get_item(key) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.is_instance_of::<PyKeyError>(document.py()) => Ok(None),
        Err(error) => Err(error),
    }
}

impl FromPyObject<'_, '_> for FileDocumentInput {
    type Error = PyErr;

    fn extract(document: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let py = document.py();
        let mime_type = optional_field(document, "mime_type")?
            .map(|value| value.extract::<String>())
            .transpose()?;
        let file = optional_field(document, "file")?.ok_or(DocumentError::MissingFile)?;
        if file.is_none() {
            return Err(DocumentError::MissingFile.into());
        }
        if file.is_instance_of::<PyString>() {
            return Err(DocumentError::BareString.into());
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
            None => Err(DocumentError::UnsupportedType(file.get_type().to_string()).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{exceptions::PyTypeError, types::PyDict};
    use rstest::rstest;

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

    enum Rejection {
        Value(&'static str),
        Type,
    }

    #[rstest]
    #[case::missing_file(c"{}", Rejection::Value("must include a 'file' field"))]
    #[case::none_file(c"{'file': None}", Rejection::Value("must include a 'file' field"))]
    #[case::none_mime_type(c"{'file': b'abc', 'mime_type': None}", Rejection::Type)]
    #[case::numeric_mime_type(c"{'file': b'abc', 'mime_type': 7}", Rejection::Type)]
    #[case::bare_str(c"{'file': 'scan.pdf'}", Rejection::Value("bare str"))]
    #[case::unsupported_type(
        c"{'file': object()}",
        Rejection::Value("Unsupported file input type")
    )]
    fn extraction_rejects_invalid_documents(
        #[case] expression: &std::ffi::CStr,
        #[case] rejection: Rejection,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let error = py
                .eval(expression, None, None)
                .unwrap()
                .extract::<FileDocumentInput>()
                .err()
                .unwrap();
            match rejection {
                Rejection::Value(fragment) => {
                    assert!(error.is_instance_of::<PyValueError>(py));
                    assert!(error.to_string().contains(fragment), "{error}");
                }
                Rejection::Type => assert!(error.is_instance_of::<PyTypeError>(py)),
            }
        });
    }

    const DEFAULTING_DICT: &std::ffi::CStr = c"class Document(dict):
    def __missing__(self, key):
        return Document()
    def __getattr__(self, name):
        return self[name]";

    const ITEM_LOOKUP_ONLY: &std::ffi::CStr = c"class Document:
    def __init__(self, items=()):
        self._items = dict(items)
    def __getitem__(self, key):
        return self._items[key]";

    #[rstest]
    #[case::without_mime_type(c"{'file': b'abc'}", None)]
    #[case::with_mime_type(c"{'file': b'abc', 'mime_type': 'image/png'}", Some("image/png"))]
    fn a_document_yields_only_the_keys_the_caller_set(
        #[values(DEFAULTING_DICT, ITEM_LOOKUP_ONLY)] class: &std::ffi::CStr,
        #[case] items: &std::ffi::CStr,
        #[case] mime_type: Option<&str>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(py, class);
            let document = locals
                .get_item("Document")
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
                .get_item("Document")
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
    fn extraction_keeps_bytes_and_their_mime_type() {
        Python::initialize();
        Python::attach(|py| {
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
