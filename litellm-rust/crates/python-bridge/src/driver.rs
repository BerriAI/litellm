use std::ffi::CString;

use pyo3::prelude::*;

const DRIVE: &str = r#"
def drive_sync(arguments):
    host = Host(arguments, False)
    while host.machine.complete() is None:
        try:
            _invoke(host.machine, host)
        except Exception as error:
            host.advance(1, error)
        except BaseException as error:
            host.advance(2, error)
        else:
            host.advance(0)
    return host.result()

async def drive_async(arguments):
    host = Host(arguments, True)
    while host.machine.complete() is None:
        try:
            awaiting, value = _invoke(host.machine, host)
            if awaiting:
                await value
        except Exception as error:
            host.advance(1, error)
        except BaseException as error:
            host.advance(2, error)
        else:
            host.advance(0)
    return host.result()
"#;

pub(crate) fn compile<'py>(
    py: Python<'py>,
    route: &str,
    host: &str,
) -> PyResult<Bound<'py, PyModule>> {
    let source = CString::new(format!("{host}\n{DRIVE}")).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("driver source contains a null byte")
    })?;
    let filename = CString::new(format!("{route}_driver.py"))
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid driver route name"))?;
    let module_name = CString::new(format!("_{route}_driver"))
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid driver route name"))?;
    PyModule::from_code(py, &source, &filename, &module_name)
}
