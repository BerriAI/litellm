use litellm_spend::CounterKey;

#[derive(Debug, thiserror::Error)]
pub enum Error<C: std::error::Error + 'static> {
    #[error("Redis spend command failed")]
    Redis(#[from] redis::RedisError),
    #[error("a buffered spend batch could not be encoded or decoded")]
    Codec(#[source] C),
    #[error("a claim record is not the JSON array of blobs this crate writes")]
    ClaimRecord(#[source] serde_json::Error),
    #[error("a claim id is not the UUID this crate writes")]
    ClaimId(#[source] uuid::Error),
    #[error("the counter naming has no counter for {0:?}")]
    Unnamed(CounterKey),
}

#[derive(Debug, thiserror::Error)]
pub enum PythonFormatError {
    #[error("buffered spend is not the JSON Python writes")]
    Json(#[from] serde_json::Error),
    #[error("member key {0:?} does not name exactly one group and one user")]
    MemberKey(String),
    #[error("id {0:?} contains the member key separator and cannot be written unambiguously")]
    SeparatorInId(String),
}
