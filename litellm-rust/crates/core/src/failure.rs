//! Where a route failure happened, which is what a host needs in order to decide what to
//! raise: a rejection before any provider I/O can be retried on another path, an upstream
//! status carries the provider's own answer, and anything after dispatch may already have
//! been billed.

use std::io;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub enum Phase<'a> {
    BeforeProvider(Rejection<'a>),
    Upstream {
        status: u16,
        body: &'a str,
        headers: &'a [(String, String)],
    },
    AfterProvider,
}

#[derive(Clone, Copy, Debug)]
pub enum Rejection<'a> {
    InvalidRequest,
    Unsupported,
    Credential,
    Unreachable,
    RequestFormat,
    FileNotFound(&'a Path),
    FileRead(&'a io::Error),
    Internal,
}

impl Rejection<'_> {
    pub fn name(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::Unsupported => "unsupported",
            Self::Credential => "credential",
            Self::Unreachable => "unreachable",
            Self::RequestFormat => "request_format",
            Self::FileNotFound(_) => "file_not_found",
            Self::FileRead(_) => "file_read",
            Self::Internal => "internal",
        }
    }
}
