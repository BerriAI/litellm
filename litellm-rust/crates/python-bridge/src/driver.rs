use std::ffi::CString;

use pyo3::prelude::*;

use crate::errors::RustBridgeDriverError;

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
    let source = CString::new(format!("{host}\n{DRIVE}"))
        .map_err(|_| RustBridgeDriverError::new_err("driver source contains a null byte"))?;
    let filename = CString::new(format!("{route}_driver.py"))
        .map_err(|_| RustBridgeDriverError::new_err("driver route name contains a null byte"))?;
    let module_name = CString::new(format!("_{route}_driver"))
        .map_err(|_| RustBridgeDriverError::new_err("driver route name contains a null byte"))?;
    PyModule::from_code(py, &source, &filename, &module_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_byte_in_route_raises_driver_error() {
        Python::initialize();
        Python::attach(|py| {
            let error = compile(py, "invalid\0route", "class Host: pass")
                .expect_err("route names containing null bytes should fail");

            assert!(error.is_instance_of::<RustBridgeDriverError>(py));
            assert_eq!(
                error.to_string(),
                "RustBridgeDriverError: driver route name contains a null byte"
            );
        });
    }

    #[test]
    fn null_byte_in_source_raises_driver_error() {
        Python::initialize();
        Python::attach(|py| {
            let error = compile(py, "test", "class Host:\0 pass")
                .expect_err("driver source containing null bytes should fail");

            assert!(error.is_instance_of::<RustBridgeDriverError>(py));
            assert_eq!(
                error.to_string(),
                "RustBridgeDriverError: driver source contains a null byte"
            );
        });
    }
}
