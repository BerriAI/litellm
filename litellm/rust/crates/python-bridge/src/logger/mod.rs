//! Delivers Rust `tracing` events to Python's `logging`, so Rust diagnostics go through the same loggers, filters
//! and handlers as Python's own.
//!
//! `host` holds every call into Python. `sink` is the sink a call scopes its Rust work to, and it logs with that
//! call's request correlation. `machine` scopes a route's execution to that sink. `unscoped` is the process-wide sink
//! for threads no call scoped, and `background` is the queue it delivers through.

mod background;
mod host;
mod levels;
mod machine;
mod sink;
mod unscoped;

pub(crate) use machine::LoggedMachine;
pub(crate) use sink::{capture, capture_call};
pub(crate) use unscoped::unscoped;
