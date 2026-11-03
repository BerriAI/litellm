use serde::{Deserialize, Serialize};

#[derive(
    Clone,
    Copy,
    Debug,
    Deserialize,
    Serialize,
    PartialEq,
    Eq,
    Hash,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
pub enum CacheType {
    #[serde(rename = "local")]
    #[strum(serialize = "local")]
    Local,
    #[serde(rename = "redis")]
    #[strum(serialize = "redis")]
    Redis,
    #[serde(rename = "redis-semantic")]
    #[strum(serialize = "redis-semantic")]
    RedisSemantic,
    #[serde(rename = "valkey-semantic")]
    #[strum(serialize = "valkey-semantic")]
    ValkeySemantic,
    #[serde(rename = "s3")]
    #[strum(serialize = "s3")]
    S3,
    #[serde(rename = "disk")]
    #[strum(serialize = "disk")]
    Disk,
    #[serde(rename = "qdrant-semantic")]
    #[strum(serialize = "qdrant-semantic")]
    QdrantSemantic,
    #[serde(rename = "azure-blob")]
    #[strum(serialize = "azure-blob")]
    AzureBlob,
    #[serde(rename = "gcs")]
    #[strum(serialize = "gcs")]
    Gcs,
}
