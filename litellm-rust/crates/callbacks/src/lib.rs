//! The contract between a native call and the host runtime that drives it.
//!
//! A host is whatever sits on the far side of the language boundary: CPython today,
//! another runtime later. Core implements [`machine::Machine`] per route and never learns
//! which host is on the other end. The machine yields [`host::HostOp`]s and, when the call
//! streams, chunks; a driver answers them, observes [`event::CallEvent`]s and may rewrite
//! the wire request before it is sent. A route's error says what kind of failure it is
//! through [`failure::Classified`], so a loop of attempts can decide without reading it.

pub mod event;
pub mod failure;
pub mod host;
pub mod layer;
pub mod machine;
pub mod route;
pub mod run;
