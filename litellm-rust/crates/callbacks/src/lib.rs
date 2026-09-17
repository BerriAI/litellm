//! The contract between a native call and the host runtime that drives it.
//!
//! A host is whatever sits on the far side of the language boundary: CPython today,
//! another runtime later. Core implements [`protocol::NativeCall`] and never learns which
//! host is on the other end; a bridge crate drives it against one.

pub mod context;
pub mod protocol;
