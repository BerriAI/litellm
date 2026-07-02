//! Realtime logging collector. Observes the realtime event stream and emits a
//! `StandardLoggingPayload` to the registered callbacks on session close.

pub mod streaming;
