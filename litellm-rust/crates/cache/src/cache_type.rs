use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
pub enum CacheType {
    #[serde(rename = "local")]
    Local,
    #[serde(rename = "redis")]
    Redis,
    #[serde(rename = "redis-semantic")]
    RedisSemantic,
    #[serde(rename = "valkey-semantic")]
    ValkeySemantic,
    #[serde(rename = "s3")]
    S3,
    #[serde(rename = "disk")]
    Disk,
    #[serde(rename = "qdrant-semantic")]
    QdrantSemantic,
    #[serde(rename = "azure-blob")]
    AzureBlob,
    #[serde(rename = "gcs")]
    Gcs,
}

impl CacheType {
    pub const ALL: [Self; 9] = [
        Self::Local,
        Self::Redis,
        Self::RedisSemantic,
        Self::ValkeySemantic,
        Self::S3,
        Self::Disk,
        Self::QdrantSemantic,
        Self::AzureBlob,
        Self::Gcs,
    ];

    pub const fn as_python_name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Redis => "redis",
            Self::RedisSemantic => "redis-semantic",
            Self::ValkeySemantic => "valkey-semantic",
            Self::S3 => "s3",
            Self::Disk => "disk",
            Self::QdrantSemantic => "qdrant-semantic",
            Self::AzureBlob => "azure-blob",
            Self::Gcs => "gcs",
        }
    }

    pub fn from_python_name(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|cache_type| cache_type.as_python_name() == value)
    }
}
