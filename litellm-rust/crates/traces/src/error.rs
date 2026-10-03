#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid OTLP trace payload")]
    InvalidPayload,
    #[error("OTLP trace payload exceeds the decoding budget")]
    TooLarge,
    #[error("OTLP token count is outside the storage range")]
    TokenCountOutOfRange,
}

#[derive(Debug, thiserror::Error)]
#[error("invalid trace query scope")]
pub struct InvalidScope;

#[derive(Debug, thiserror::Error)]
#[error("unknown ClickHouse read query")]
pub struct InvalidQuery;
