use crate::{
    Error, Unavailable,
    host::Rows,
    jwt::{JwtIdentity, MappedJwt},
    key::{KeyContext, KeyIdentity},
};

#[derive(Clone, Debug, PartialEq)]
pub enum Identity<R: Rows> {
    Key(KeyIdentity<R>),
    /// A JWT mapped onto a virtual key that then passed the key checks.
    MappedJwtKey(MappedJwt<R::User>, Box<KeyContext<R>>),
    /// A proxy admin's JWT, with the rows the host converts.
    AdminJwt(Box<JwtIdentity<R>>),
    /// The token the host finished for a JWT, or for the key auto-registered for it.
    FinishedJwt(R::Key),
    /// The database could not be reached and `allow_requests_on_db_unavailable` lets the request
    /// through as a restricted internal user.
    DbUnavailable(Unavailable),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome<R: Rows> {
    Authenticated(Identity<R>),
    /// A denial, never a host failure, with the key the request had resolved before it.
    Denied(Error, Option<R::Key>),
}
