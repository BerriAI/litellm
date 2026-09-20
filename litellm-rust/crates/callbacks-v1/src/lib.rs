//! The runtime-neutral v1 callback contract: envelopes and their sequencing, credential
//! redaction, handler selection and validated wire patches. No PyO3 and no registry; a
//! language host marshals these values to its own subscribers.

mod envelope;
mod patch;
mod policy;
mod redact;

pub use envelope::{
    CallFacts, Envelope, ErrorFacts, Event, EventKind, Origin, RequestFacts, SCHEMA_V1, Sequencer,
    TimingFacts,
};
pub use patch::{HeaderPatch, PatchError, WirePatch, apply};
pub use policy::{ExecutionMode, HandlerSelection, select_handler};
pub use redact::REDACTED;
