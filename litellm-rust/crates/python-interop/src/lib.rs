//! pyo3 plumbing shared by the crates that speak Python: value marshalling and
//! interpreter detachment. Everything here is Python-specific by construction, so a
//! second host language needs its own bridge rather than a sibling of this crate.

mod gil;
mod marshal;

pub use gil::{release_count, release_gil};
pub use marshal::{Pythonized, from_py, from_py_argument, panic_to_pyerr, to_py};
