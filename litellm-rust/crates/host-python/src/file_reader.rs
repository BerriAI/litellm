//! A caller's file-like object: anything with a callable `read`, kept as a handle and read
//! once, on the host's thread, into bytes Rust owns.

use bytes::Bytes;
use pyo3::{
    exceptions::PyTypeError,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    pybacked::PyBackedBytes,
    types::{PyBytes, PyString},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileContent {
    pub bytes: Bytes,
    pub file_name: Option<String>,
}

#[derive(Debug)]
pub struct PythonFileReader {
    reader: Py<PyAny>,
    name: Option<String>,
}

impl PythonFileReader {
    /// `None` when `file` has no callable `read`. The object's `name` is read now, its
    /// contents only on [`read`](Self::read).
    pub fn from_file_like(file: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        let reader = file
            .getattr_opt("read")?
            .filter(|value| value.is_callable());
        let Some(reader) = reader else {
            return Ok(None);
        };
        let name = file
            .getattr_opt("name")?
            .filter(|value| !value.is_none())
            .map(|value| value.extract::<String>())
            .transpose()?;
        Ok(Some(Self {
            reader: reader.unbind(),
            name,
        }))
    }

    pub fn read(&self, py: Python<'_>) -> PyResult<FileContent> {
        let value = self.reader.bind(py).call0()?;
        let bytes = if value.is_instance_of::<PyString>() {
            Bytes::from(value.extract::<String>()?)
        } else if value.is_instance_of::<PyBytes>() {
            py_bytes(&value)?
        } else {
            return Err(PyTypeError::new_err(format!(
                "file read must return bytes or str, got {}",
                value.get_type(),
            )));
        };
        Ok(FileContent {
            bytes,
            file_name: self.name.clone(),
        })
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.reader)
    }
}

/// An exact `bytes` object is shared without copying and keeps the Python object alive;
/// a `bytes` subclass is copied.
pub fn py_bytes(value: &Bound<'_, PyAny>) -> PyResult<Bytes> {
    if value.is_exact_instance_of::<PyBytes>() {
        return Ok(Bytes::from_owner(value.extract::<PyBackedBytes>()?));
    }
    Ok(Bytes::copy_from_slice(
        value.extract::<PyBackedBytes>()?.as_ref(),
    ))
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

    fn reader<'py>(locals: &Bound<'py, PyDict>, name: &str) -> PythonFileReader {
        PythonFileReader::from_file_like(&locals.get_item(name).unwrap().unwrap())
            .unwrap()
            .unwrap()
    }

    #[test]
    fn objects_without_a_callable_read_are_not_readers() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
class Attribute:
    read = 'not callable'
plain = object()
attribute = Attribute()
",
            );
            for name in ["plain", "attribute"] {
                let file = locals.get_item(name).unwrap().unwrap();
                assert!(PythonFileReader::from_file_like(&file).unwrap().is_none());
            }
        });
    }

    #[test]
    fn the_name_is_taken_up_front_and_the_contents_only_on_read() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
class Reader:
    name = 'scan.png'
    def __init__(self):
        self.reads = 0
    def read(self):
        self.reads += 1
        return b'abc'
file = Reader()
",
            );
            let reads = || {
                locals
                    .get_item("file")
                    .unwrap()
                    .unwrap()
                    .getattr("reads")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap()
            };
            let file = reader(&locals, "file");
            assert_eq!(reads(), 0);
            let content = file.read(py).unwrap();
            assert_eq!(reads(), 1);
            assert_eq!(
                content,
                FileContent {
                    bytes: b"abc".as_slice().into(),
                    file_name: Some("scan.png".into()),
                }
            );
        });
    }

    #[test]
    fn read_results_are_normalized_and_exceptions_keep_their_identity() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
failure = KeyError('reader failed')
class Raising:
    def read(self):
        raise failure
class Text:
    def read(self):
        return 'héllo'
class Wrong:
    def read(self):
        return 7
raising = Raising()
text = Text()
wrong = Wrong()
",
            );
            let error = reader(&locals, "raising").read(py).unwrap_err();
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
            assert_eq!(
                reader(&locals, "text").read(py).unwrap().bytes.as_ref(),
                "héllo".as_bytes()
            );
            let error = reader(&locals, "wrong").read(py).unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
            assert!(error.to_string().contains("bytes or str"));
        });
    }

    #[rstest::rstest]
    #[case::read("read")]
    #[case::name("name")]
    fn attribute_failures_keep_their_identity(#[case] attribute: &str) {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
failure = LookupError('file property failed')
class File:
    def __getattribute__(self, name):
        if name == attribute:
            raise failure
        return super().__getattribute__(name)
    name = 'scan.pdf'
    def read(self):
        return b'abc'
file = File()
",
            );
            locals.set_item("attribute", attribute).unwrap();
            let error =
                PythonFileReader::from_file_like(&locals.get_item("file").unwrap().unwrap())
                    .unwrap_err();
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
            (py_bytes(value.as_any()).unwrap(), pointer)
        });
        assert_eq!(bytes.as_ptr() as usize, pointer);
        assert_eq!(bytes.as_ref(), b"document bytes");
    }
}
