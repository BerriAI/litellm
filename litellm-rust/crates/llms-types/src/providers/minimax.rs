use serde_json::{Map, Value};

use crate::formats::messages::CacheControl;

#[macro_rules_attribute::apply(crate::wire_type)]
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
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct MinimaxMediaBlock {
    pub source: MinimaxMediaSource,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MinimaxMediaSource {
    Base64 {
        media_type: String,
        data: String,
        #[serde(flatten)]
        options: MinimaxMediaOptions,
    },
    Url {
        url: String,
        #[serde(flatten)]
        options: MinimaxMediaOptions,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct MinimaxMediaOptions {
    pub detail: Option<MinimaxMediaDetail>,
    pub fps: Option<serde_json::Number>,
    pub max_long_side_pixel: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum MinimaxMediaDetail {
    Low,
    Default,
    High,
}
