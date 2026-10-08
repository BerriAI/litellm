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
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum CacheType {
    Local,
    Redis,
    RedisSemantic,
    ValkeySemantic,
    S3,
    Disk,
    QdrantSemantic,
    AzureBlob,
    Gcs,
}

impl CacheType {
    pub fn as_python_name(self) -> &'static str {
        self.into()
    }

    pub fn from_python_name(value: &str) -> Option<Self> {
        value.parse().ok()
    }
}
