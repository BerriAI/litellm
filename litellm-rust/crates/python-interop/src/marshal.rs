use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

use pyo3::exceptions::PyValueError;
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;

pub fn from_py<T>(value: &Bound<'_, PyAny>) -> PyResult<T>
where
    T: DeserializeOwned,
{
    pythonize::depythonize(value).map_err(|error| PyValueError::new_err(error.to_string()))
}

pub fn to_py<T>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>>
where
    T: Serialize + ?Sized,
{
    pythonize::pythonize(py, value)
        .map(Bound::unbind)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub struct Pythonized<T>(pub T);

impl<'py, T> IntoPyObject<'py> for Pythonized<T>
where
    T: Serialize,
{
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        catch_unwind(AssertUnwindSafe(|| pythonize::pythonize(py, &self.0)))
            .map_err(panic_to_pyerr)?
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

pub fn panic_to_pyerr(payload: Box<dyn Any + Send>) -> PyErr {
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("panic from Rust code");
    PanicException::new_err(message.to_string())
}

#[cfg(test)]
mod tests {
    use serde::Serializer;

    use super::*;

    struct PanickingSerializer;

    impl Serialize for PanickingSerializer {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            panic!("serializer panicked")
        }
    }

    #[test]
    fn pythonized_converts_on_the_attached_thread() {
        Python::initialize();
        Python::attach(|py| {
            let value: Vec<i32> = Pythonized(vec![1, 2, 3])
                .into_pyobject(py)
                .and_then(|value| value.extract())
                .expect("value should convert");
            assert_eq!(value, vec![1, 2, 3]);
        });
    }

    #[test]
    fn pythonized_maps_serializer_panics_to_a_base_exception() {
        Python::initialize();
        Python::attach(|py| {
            let error = Pythonized(PanickingSerializer)
                .into_pyobject(py)
                .expect_err("serializer panic should become a Python exception");
            assert!(error.is_instance_of::<PanicException>(py));
            assert_eq!(error.to_string(), "PanicException: serializer panicked");
        });
    }
}
