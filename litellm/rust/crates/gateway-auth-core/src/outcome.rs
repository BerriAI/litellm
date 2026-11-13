use crate::{
    Error,
    credential::jwt::{JwtIdentity, MappedJwt},
    credential::key::{KeyContext, KeyIdentity},
    host::Rows,
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
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome<R: Rows> {
    Authenticated(Identity<R>),
    /// A denial, never a host failure, with the key the request had resolved before it.
    Denied(Error, Option<R::Key>),
}
