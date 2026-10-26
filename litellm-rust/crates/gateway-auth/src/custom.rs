use std::{future::Future, pin::Pin};

use axum::http::request::Parts;
use litellm_auth_types::SecretValue;

use crate::{Caller, Error};

pub enum CustomOutcome {
    Caller(Caller),
    Rekey(SecretValue),
    Fallthrough,
    Reject(Error),
}

pub trait CustomAuth: Send + Sync {
    fn authenticate<'a>(
        &'a self,
        request: &'a Parts,
        key: Option<&'a SecretValue>,
    ) -> Pin<Box<dyn Future<Output = CustomOutcome> + Send + 'a>>;
}
