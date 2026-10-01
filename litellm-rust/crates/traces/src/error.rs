#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("invalid OTLP trace payload")]
    InvalidPayload,
    #[error("OTLP trace payload exceeds the decompressed size limit")]
    TooLarge,
}
