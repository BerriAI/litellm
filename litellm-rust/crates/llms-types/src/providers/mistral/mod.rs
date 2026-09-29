pub mod audio;
pub mod batches;
pub mod chat;
pub mod embeddings;
pub mod files;
pub mod fim;
pub mod ocr;

use serde_json::{Map, Value};

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MistralUsage {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub prompt_tokens: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub completion_tokens: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub total_tokens: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub prompt_audio_seconds: Option<Option<u64>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}
