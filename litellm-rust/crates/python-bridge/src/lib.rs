#![recursion_limit = "256"]

mod arguments;
mod callbacks;
mod diagnostics;
mod driver;
mod errors;
mod marshal;
mod retained;
mod routes;
mod runtime;
#[cfg(feature = "trace-parity")]
mod trace_parity;

use pyo3::prelude::*;

#[pymodule(gil_used = true)]
pub mod _native {
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        super::errors::register(module)?;
        super::routes::register(module)?;
        super::diagnostics::register(module)
    }
}
