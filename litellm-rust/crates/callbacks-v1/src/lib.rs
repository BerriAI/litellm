//! The runtime-neutral v1 callback contract: envelopes, credential redaction, handler
//! selection, validated wire patches, and the per-call [`CallSession`] that says which
//! envelope goes to whom and folds interceptor patches. No PyO3, no registry and no I/O; a
//! host projects its own values into facts and runs its own subscribers.

mod envelope;
mod patch;
mod policy;
mod redact;
mod session;

pub use envelope::{
    CallFacts, Envelope, ErrorFacts, Event, EventKind, Origin, RequestFacts, SCHEMA_V1, TimingFacts,
};
pub use patch::{HeaderPatch, PatchError, WirePatch, apply};
pub use policy::{ExecutionMode, HandlerSelection, select_handler};
pub use session::{CallSession, Emission, Interception, Subscription, Turn};
