//! The contract between a native call and the host runtime that drives it.
//!
//! A host is whatever sits on the far side of the language boundary: CPython today,
//! another runtime later. Core runs each route on a [`machine::CallMachine`] and never learns
//! which host is on the other end. The machine yields [`host::HostOp`]s; a driver answers
//! each through the typed [`host::Reply`] it carries, observes [`event::CallEvent`]s and
//! may rewrite the wire request before it is sent.

mod error;
pub use error::MachineFault;
pub mod event;
pub mod hooks;
pub mod host;
pub mod machine;
pub mod protocol;
pub mod run;
