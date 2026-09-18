//! Failures raised by a caller-supplied Python callable.

use pyo3::exceptions::{PyException, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Reports a caller-supplied callable's failure under `template`, a Python format string
/// with one field for the original exception, while leaving alone the failures a caller
/// can already read: a `TypeError`, so a rejected return value is not reported twice, and
/// anything that is not a `PyException`, a cancellation for example. Everything else
/// becomes a `RuntimeError` carrying the original as both its `__cause__` and its
/// `__context__`, with the message rendered by Python so the exception's own `__format__`
/// is honored. A `__format__` that raises surfaces as that failure instead, with the
/// original attached as its context.
pub fn wrap_failure<T>(py: Python<'_>, template: &str, result: PyResult<T>) -> PyResult<T> {
    result.map_err(|error| {
        if error.is_instance_of::<PyTypeError>(py) || !error.is_instance_of::<PyException>(py) {
            return error;
        }
        match PyString::new(py, template).call_method1("format", (error.value(py),)) {
            Ok(message) => {
                let wrapped = PyRuntimeError::new_err(message.unbind());
                wrapped.set_context(py, Some(error.clone_ref(py)));
                wrapped.set_cause(py, Some(error));
                wrapped
            }
            Err(format_error) => {
                format_error.set_context(py, Some(error));
                format_error
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use pyo3::types::PyDict;

    use super::*;

    const TEMPLATE: &str = "Failed to reach the caller: {}";

    fn raised<'py>(locals: &Bound<'py, PyDict>, name: &str) -> Bound<'py, PyAny> {
        locals.get_item(name).unwrap().unwrap()
    }

    fn failure<'py>(error: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        Err(PyErr::from_value(error.clone()))
    }

    #[test]
    fn only_ordinary_exceptions_are_reported_under_the_template() {
        crate::initialize_python();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
class CallerError(Exception):
    def __format__(self, specification):
        return 'unavailable'
ordinary = CallerError('must use __format__')
type_error = TypeError('signature')
abort = KeyboardInterrupt('cancelled')
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();

            let original = raised(&locals, "ordinary");
            let wrapped = wrap_failure(py, TEMPLATE, failure(&original)).unwrap_err();
            assert!(wrapped.is_instance_of::<PyRuntimeError>(py));
            assert!(wrapped.cause(py).unwrap().value(py).is(&original));
            assert!(
                wrapped
                    .value(py)
                    .getattr("__context__")
                    .unwrap()
                    .is(&original)
            );
            assert_eq!(
                wrapped.value(py).str().unwrap().to_str().unwrap(),
                "Failed to reach the caller: unavailable"
            );

            for name in ["type_error", "abort"] {
                let original = raised(&locals, name);
                let error = wrap_failure(py, TEMPLATE, failure(&original)).unwrap_err();
                assert!(error.value(py).is(&original));
            }
        });
    }

    #[test]
    fn a_raising_format_surfaces_instead_of_the_report_and_keeps_the_original_as_context() {
        crate::initialize_python();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
class Unformattable(Exception):
    def __format__(self, specification):
        raise ValueError('formatting failed')
original = Unformattable('cannot render')
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();

            let original = raised(&locals, "original");
            let error = wrap_failure(py, TEMPLATE, failure(&original)).unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(
                error
                    .value(py)
                    .getattr("__context__")
                    .unwrap()
                    .is(&original)
            );
        });
    }

    #[test]
    fn successful_results_pass_through_untouched() {
        crate::initialize_python();
        Python::attach(|py| {
            assert_eq!(wrap_failure(py, TEMPLATE, Ok(7)).unwrap(), 7);
        });
    }
}
