use serde_json::{Map, Value};

use crate::formats::messages::{CacheControl, Citations, ContentSource};

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ImageUrl {
        image_url: ChatMediaUrl,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    VideoUrl {
        video_url: ChatMediaUrl,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    InputAudio {
        input_audio: ChatInputAudio,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    File {
        file: Box<ChatFile>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Document {
        source: Box<ContentSource>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<Citations>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Refusal {
        refusal: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum ChatMediaUrl {
    Url(String),
    Parameters(Box<ChatMediaUrlParameters>),
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatMediaUrlParameters {
    pub url: Option<String>,

    pub detail: Option<String>,

    pub format: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatInputAudio {
    pub data: Option<String>,

    pub format: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatFile {
    pub file_data: Option<String>,

    pub file_id: Option<String>,

    pub filename: Option<String>,

    pub format: Option<String>,

    pub detail: Option<String>,

    pub video_metadata: Option<ChatVideoMetadata>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatVideoMetadata {
    pub fps: Option<serde_json::Number>,

    pub start_offset: Option<String>,

    pub end_offset: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatLogprobs {
    pub content: Option<Vec<ChatTokenLogprob>>,

    pub refusal: Option<Vec<ChatTokenLogprob>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatTokenLogprob {
    pub token: Option<String>,

    pub logprob: Option<serde_json::Number>,

    pub bytes: Option<Vec<u8>>,

    pub top_logprobs: Option<Vec<ChatTopLogprob>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatTopLogprob {
    pub token: Option<String>,

    pub logprob: Option<serde_json::Number>,

    pub bytes: Option<Vec<u8>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
