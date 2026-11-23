//! Each check reads the config the native app was built from and one part of a request, and names
//! why the native app would not answer that request as Python does, or `None`. The checks run in
//! three stages, each named for what its checks read and each in order, cheapest first. `mod.rs`
//! authenticates the caller before `CALLER` and reads the body before `BODY`, so neither happens
//! for a request an earlier stage sent to Python. A request goes to Python only over a check's
//! reason.

#[macro_use]
mod fields;

mod body;
mod caller;
mod config;
mod head;
mod model;
mod router;

use axum::http::request::Parts;
use litellm_gateway_auth::Authenticated;
use serde_json::{Map, Value};

pub(super) use model::alias;

use super::{Admit, Refused};
use crate::host::NativeRows;

/// Why one request goes to Python: the reason the client is told, and the setting or trait behind
/// it, which only the log names.
#[derive(Debug, PartialEq, Eq)]
pub struct Reason {
    pub why: Refused,
    pub over: Box<str>,
}

impl Reason {
    pub fn new(why: Refused, over: impl Into<Box<str>>) -> Self {
        Self {
            why,
            over: over.into(),
        }
    }
}

pub(super) type Authentication = Result<Authenticated<NativeRows>, litellm_gateway_auth::Error>;
pub(super) type Fields = Map<String, Value>;

pub(super) const HEAD: &[fn(&Admit, &Parts) -> Option<Reason>] =
    &[config::settings, head::route, head::headers];
pub(super) const CALLER: &[fn(&Admit, &Authentication) -> Option<Reason>] = &[caller::caller];
pub(super) const BODY: &[fn(&Admit, &Fields) -> Option<Reason>] = &[body::fields, model::model];
