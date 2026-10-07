use serde_json::{Map, Value};

use crate::recognized::Recognized;
use crate::serde_compat::deserialize_present;

use crate::formats::messages::{CacheControl, Citations, ContentSource};

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<Recognized<ChatContentPart>>),
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    Text {
        text: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present"
        )]
        cache_control: Option<Recognized<CacheControl>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ImageUrl {
        image_url: Recognized<ChatMediaUrl>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    VideoUrl {
        video_url: Recognized<ChatMediaUrl>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    InputAudio {
        input_audio: Recognized<ChatInputAudio>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    File {
        file: Box<Recognized<ChatFile>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Document {
        source: Box<Recognized<ContentSource>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present"
        )]
        title: Option<Recognized<String>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present"
        )]
        context: Option<Recognized<String>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present"
        )]
        citations: Option<Recognized<Citations>>,
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
    #[serde(default, deserialize_with = "deserialize_present")]
    pub url: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub detail: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub format: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatInputAudio {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub data: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub format: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatFile {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub file_data: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub file_id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub filename: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub format: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub detail: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub video_metadata: Option<Recognized<ChatVideoMetadata>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatVideoMetadata {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub fps: Option<Recognized<serde_json::Number>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub start_offset: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub end_offset: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatLogprobs {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub content: Option<Recognized<Vec<Recognized<ChatTokenLogprob>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub refusal: Option<Recognized<Vec<Recognized<ChatTokenLogprob>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatTokenLogprob {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub token: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub logprob: Option<Recognized<serde_json::Number>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub bytes: Option<Recognized<Vec<u8>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub top_logprobs: Option<Recognized<Vec<Recognized<ChatTopLogprob>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ChatTopLogprob {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub token: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub logprob: Option<Recognized<serde_json::Number>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub bytes: Option<Recognized<Vec<u8>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Recognized<ChatContentPart> {
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Known(ChatContentPart::Text { text, .. }) => Some(text),
            Self::Known(
                ChatContentPart::ImageUrl { extra, .. }
                | ChatContentPart::VideoUrl { extra, .. }
                | ChatContentPart::InputAudio { extra, .. }
                | ChatContentPart::File { extra, .. }
                | ChatContentPart::Document { extra, .. }
                | ChatContentPart::Refusal { extra, .. },
            ) => extra.get("text").and_then(Value::as_str),
            Self::Unrecognized(value) => value.get("text").and_then(Value::as_str),
        }
    }

    pub fn plain_text(&self) -> Option<&str> {
        match self {
            Self::Known(ChatContentPart::Text {
                text,
                cache_control: None,
                extra,
            }) if extra.is_empty() => Some(text),
            _ => None,
        }
    }
}
