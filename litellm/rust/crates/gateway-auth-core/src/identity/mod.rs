//! Who is acting once the credential is genuine: a key's owner, team and project, or the caller a
//! JWT names and the team, user and organization behind it.

mod caller;
pub mod jwt;
pub mod key;
mod principal;

pub use caller::{Caller, KeyCaller, NotACaller, Scope};
pub(crate) use principal::{CallerKind, ClaimedCaller, Principal};
