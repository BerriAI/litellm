//! Delivers Rust `tracing` events to Python's `logging`, so Rust diagnostics go through the same loggers, filters
//! and handlers as Python's own.
//!
//! `process` is the one sink every event goes through, and it gives each call the logger that carries its request
//! ids. `host` holds every call into Python, `route` picks the Python logger for a Rust target, and `levels` keeps
//! the level each of those loggers takes. `background` is the queue that keeps Python handlers off the async
//! workers, and `machine` runs a route's execution inside its call's logger.

mod background;
mod host;
mod levels;
mod machine;
mod process;
mod route;

pub(crate) use machine::LoggedMachine;
pub(crate) use process::{capture, process};
