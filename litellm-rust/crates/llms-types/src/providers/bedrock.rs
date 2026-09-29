use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::recognized::{Recognized, deserialize_present};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockInvocationMetrics {
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub input_token_count: Option<Recognized<u64>>,
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_token_count: Option<Recognized<u64>>,
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_read_input_token_count: Option<Recognized<u64>>,
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_write_input_token_count: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
