//! Who is acting once the credential is genuine: a key's owner, team and project, or the caller a
//! JWT names and the team, user and organization behind it.

pub mod jwt;
pub mod key;
mod principal;

pub(crate) use principal::{Caller, CallerKind, Principal};
