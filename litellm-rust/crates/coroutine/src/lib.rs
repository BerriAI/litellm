//! Async coroutines on stable Rust whose every yield carries its own typed [`Reply`].
//! See `AGENTS.md` for the requirement, the alternatives and the contracts.

mod co;
mod coroutine;
mod error;
mod reply;

pub use co::Co;
pub use coroutine::{Coroutine, CoroutineState};
pub use error::{Abandoned, ResumeError};
pub use reply::{Answer, Reply, reply};
