#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("invalid OTLP trace payload")]
    InvalidPayload,
    #[error("OTLP trace payload exceeds the decoding budget")]
    TooLarge,
    #[error("OTLP token count is outside the storage range")]
    TokenCountOutOfRange,
}
