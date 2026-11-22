use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("expected s3://bucket/key")]
    InvalidUri,
}
