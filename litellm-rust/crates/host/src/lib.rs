//! The contract between a native call and the host runtime that drives it.
//!
//! A host is whatever sits on the far side of the language boundary: CPython today,
//! another runtime later. Core runs each route on a [`machine::CallMachine`] and never learns
//! which host is on the other end. The machine yields [`protocol::HostRequest`]s; a driver answers
//! each through the typed [`protocol::Reply`] it carries, observes [`lifecycle::CallEvent`]s and
//! may rewrite the wire request before it is sent.

pub mod call;
pub mod interceptors;
pub mod lifecycle;
pub mod machine;
pub mod observation;
pub mod protocol;
