use thiserror::Error;

#[derive(Debug, Error)]
#[error("Watsonx IAM token endpoint returned HTTP {status}")]
pub(crate) struct IamHttpStatus {
    pub(crate) status: u16,
}
