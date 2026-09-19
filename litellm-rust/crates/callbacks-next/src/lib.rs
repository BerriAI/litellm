mod adapter;
mod call;
mod envelope;
mod next_python;
mod patch;
mod redact;
mod registry;

pub use call::{NextSurface, run_next_call};

#[cfg(test)]
#[path = "../tests/lifecycle.rs"]
mod lifecycle_tests;

#[cfg(test)]
#[path = "../tests/support.rs"]
mod test_support;
