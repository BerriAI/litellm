#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{setting} is not a URL the database pool can read")]
    Url { setting: &'static str },
    #[error("{setting} uses unsupported scheme '{scheme}', the database must be PostgreSQL")]
    Scheme {
        setting: &'static str,
        scheme: String,
    },
    #[error("{setting} must name a host and a user")]
    Incomplete { setting: &'static str },
    #[error("invalid database setting {setting}: expected {expected}")]
    Invalid { setting: String, expected: Expected },
    #[error("{variable} asks for a minted database token, which the Rust gateway cannot mint yet")]
    TokenAuth { variable: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display)]
pub enum Expected {
    #[strum(serialize = "a positive integer")]
    PositiveInteger,
    #[strum(serialize = "a non-negative integer")]
    NonNegativeInteger,
    #[strum(serialize = "a number of seconds")]
    Seconds,
    #[strum(serialize = "a boolean")]
    Boolean,
    #[strum(serialize = "a port number")]
    Port,
}
