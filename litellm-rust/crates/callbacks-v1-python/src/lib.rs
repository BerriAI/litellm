mod adapter;
mod call;
mod python;
mod registry;

pub use call::{V1PythonSurface, run_v1_python_call};

#[cfg(test)]
#[path = "../tests/lifecycle.rs"]
mod lifecycle_tests;

#[cfg(test)]
#[path = "../tests/support.rs"]
mod test_support;
