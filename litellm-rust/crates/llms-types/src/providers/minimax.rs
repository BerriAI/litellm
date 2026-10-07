use serde_json::{Map, Value};

use crate::formats::messages::CacheControl;
use crate::recognized::Recognized;
use crate::serde_compat::deserialize_present;

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MinimaxMessagesContentBlock {
    Image(MinimaxMediaBlock),
    Video(MinimaxMediaBlock),
    MidConvSystem {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct MinimaxMediaBlock {
    pub source: Recognized<MinimaxMediaSource>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cache_control: Option<Recognized<CacheControl>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct MinimaxMediaSource {
    #[serde(rename = "type")]
    pub source_type: MinimaxMediaSourceType,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub media_type: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub data: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub url: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub detail: Option<Recognized<MinimaxMediaDetail>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub fps: Option<Recognized<serde_json::Number>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub max_long_side_pixel: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum MinimaxMediaSourceType {
    Base64,
    Url,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum MinimaxMediaDetail {
    Low,
    Default,
    High,
}
