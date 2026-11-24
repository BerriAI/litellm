use crate::{
    Error,
    authn::jwt::MappedJwt,
    host::Rows,
    identity::{
        jwt::{AutoRegisteredJwt, JwtIdentity},
        key::{KeyContext, KeyIdentity},
    },
};

#[derive(Clone, Debug, PartialEq)]
pub enum Identity<R: Rows> {
    Key(KeyIdentity<R>),
    /// A JWT mapped onto a virtual key that then passed the key checks.
    MappedJwtKey(MappedJwt<R::User>, Box<KeyContext<R>>),
    /// A JWT that acts as itself, with the rows the host converts.
    Jwt(Box<JwtIdentity<R>>),
    /// A key auto-registered for a JWT that stands in for it without the virtual-key checks.
    AutoRegisteredJwt(Box<AutoRegisteredJwt<R>>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome<R: Rows> {
    Authenticated(Identity<R>),
    /// A denial, never a host failure, with the key the request had resolved before it.
    Denied(Error, Option<R::Key>),
}

/// An error, with the key the request had resolved before it, which the host labels its failure with.
#[derive(Clone, Debug, PartialEq)]
pub struct Refused<R: Rows> {
    pub error: Box<Error>,
    pub key: Option<R::Key>,
}

impl<R: Rows> Refused<R> {
    pub(crate) fn new(error: impl Into<Error>, key: Option<&R::Key>) -> Self {
        Self {
            error: Box::new(error.into()),
            key: key.cloned(),
        }
    }
}

impl<R: Rows> From<Error> for Refused<R> {
    fn from(error: Error) -> Self {
        Self {
            error: Box::new(error),
            key: None,
        }
    }
}

pub type KeyOutcome<R> = Result<KeyIdentity<R>, Refused<R>>;
