//! Test support for machine stacks, in the spirit of `tower-test`: a route whose ops are
//! strings, a machine that plays a script, a host that drains a stack while recording every
//! op it answers, and assertions over that trace. Behavior tests for a stack are written as
//! "given these attempts and this host, the trace is exactly this and the outcome is that".

mod host;
mod route;
mod scripted;
mod trace;

pub use host::{Answer, Answers, RecordingHost, WithOuter, drain};
pub use route::TestRoute;
pub use scripted::{Script, Scripted};
pub use trace::{Observed, Trace};
