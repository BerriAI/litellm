//! The out-of-process host of the v1 callback contract: the gateway's side of the socket to
//! an extension host worker (`python -m litellm.callbacks_v1.host`). The gateway never loads
//! a callback; the worker reports its subscriptions, and from then on this crate sends it
//! what a [`CallSession`](litellm_callbacks_v1::CallSession) says to deliver and brings
//! back interceptor patches for the session to validate.
//!
//! [`protocol`] is the wire, pinned by `golden/` on both sides. [`Worker`] is the blocking
//! reference client: what a conforming gateway does, in the simplest form that can be
//! tested end to end. The production host (async, supervised, queued) reuses the protocol
//! and replaces the client.

pub mod protocol;
mod worker;

pub use worker::{ExtHostError, Report, Worker};
