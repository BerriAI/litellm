use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct ConverseResponse {
    pub output: ConverseOutput,
    #[serde(default)]
    pub usage: ConverseUsage,
    #[serde(rename = "stopReason")]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ConverseOutput {
    pub message: ConverseMessage,
}

#[derive(Debug, Deserialize)]
pub struct ConverseMessage {
    pub content: Vec<ConverseContentBlock>,
}

#[derive(Debug)]
pub enum ConverseContentBlock {
    Text { text: String },
    Other,
}

impl<'de> Deserialize<'de> for ConverseContentBlock {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let Some(text) = value.get("text") else {
            return Ok(Self::Other);
        };
        let text = text.as_str().ok_or_else(|| {
            serde::de::Error::custom("invalid type for `text`, expected a string")
        })?;
        Ok(Self::Text {
            text: text.to_owned(),
        })
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
    #[serde(default)]
    pub cache_write_input_tokens: u64,
    pub total_tokens: Option<u64>,
}

impl ConverseResponse {
    pub fn content_text(&self) -> String {
        self.output
            .message
            .content
            .iter()
            .filter_map(|block| match block {
                ConverseContentBlock::Text { text } => Some(text.as_str()),
                ConverseContentBlock::Other => None,
            })
            .collect()
    }

    pub fn message_content_is_non_text(&self) -> bool {
        self.output
            .message
            .content
            .iter()
            .any(|block| matches!(block, ConverseContentBlock::Other))
    }
}
