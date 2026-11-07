use litellm_error::Source;

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid ClickHouse insert row")]
    InvalidRow(#[source] Source),
    #[error("{0} must be a positive integer")]
    InvalidLimit(&'static str),
    #[error("invalid ClickHouse insert table")]
    InvalidTable,
    #[error("invalid ClickHouse HTTP URL")]
    InvalidUrl(#[source] url::ParseError),
    #[error("ClickHouse URL must use http or https and name a host")]
    UnsupportedUrl,
    #[error("database must be a nonempty SQL identifier and retention must be positive")]
    InvalidSchema,
    #[error("SQL query must not be empty")]
    EmptySql,
    #[error("invalid ClickHouse query parameters")]
    InvalidParameters(#[source] Source),
    #[error("ClickHouse query failed with HTTP status {0}")]
    QueryFailed(u16),
    #[error("ClickHouse insert failed with HTTP status {0}")]
    InsertFailed(u16),
    #[error("ClickHouse insert exceeds the encoded size limit")]
    InsertTooLarge,
    #[error("ClickHouse schema setup failed with HTTP status {0}")]
    SchemaFailed(u16),
    #[error("ClickHouse migrations are forward-only")]
    Irreversible,
    #[error("ClickHouse query exceeded the response size limit")]
    ResponseTooLarge,
    #[error("ClickHouse returned an invalid response")]
    InvalidResponse(#[source] Source),
    #[error("ClickHouse returned a failed or unexpected response")]
    UnexpectedResponse,
    #[error("ClickHouse transport failed")]
    Transport(#[from] litellm_http::Error),
}

impl Error {
    pub fn is_request(&self) -> bool {
        match self {
            Self::InvalidRow(_)
            | Self::InvalidLimit(_)
            | Self::InvalidTable
            | Self::InvalidSchema
            | Self::EmptySql
            | Self::InvalidParameters(_) => true,
            Self::InvalidUrl(_)
            | Self::UnsupportedUrl
            | Self::QueryFailed(_)
            | Self::InsertFailed(_)
            | Self::InsertTooLarge
            | Self::SchemaFailed(_)
            | Self::Irreversible
            | Self::ResponseTooLarge
            | Self::InvalidResponse(_)
            | Self::UnexpectedResponse
            | Self::Transport(_) => false,
        }
    }

    pub fn is_insert_too_large(&self) -> bool {
        match self {
            Self::InsertTooLarge => true,
            Self::InvalidRow(_)
            | Self::InvalidLimit(_)
            | Self::InvalidTable
            | Self::InvalidUrl(_)
            | Self::UnsupportedUrl
            | Self::InvalidSchema
            | Self::EmptySql
            | Self::InvalidParameters(_)
            | Self::QueryFailed(_)
            | Self::InsertFailed(_)
            | Self::SchemaFailed(_)
            | Self::Irreversible
            | Self::ResponseTooLarge
            | Self::InvalidResponse(_)
            | Self::UnexpectedResponse
            | Self::Transport(_) => false,
        }
    }
}
