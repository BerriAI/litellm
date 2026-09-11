use std::cell::Cell;

use pyo3::prelude::*;

thread_local! {
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

pub fn in_callback() -> bool {
    IN_CALLBACK.get()
}

pub fn invoke_callback(callback: &Py<PyAny>) -> PyResult<Py<PyAny>> {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_CALLBACK.set(self.0);
        }
    }
    let _restore = Restore(IN_CALLBACK.replace(true));
    Python::attach(|py| callback.call0(py))
}
