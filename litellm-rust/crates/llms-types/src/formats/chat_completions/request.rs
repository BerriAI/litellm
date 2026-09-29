use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::recognized::Recognized;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<Recognized<ChatContentPart>>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    Text {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ImageUrl {
        image_url: ChatMediaUrl,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    VideoUrl {
        video_url: ChatVideoUrl,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

impl ChatContentPart {
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text { text, .. } => Some(text),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatMediaUrl {
    pub url: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatVideoUrl {
    Url(String),
    Options(ChatMediaUrl),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatMessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
