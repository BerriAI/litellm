use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::ChatCompletionsUsage;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionToolCallFunctionChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionToolCallChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ChatCompletionToolCallFunctionChunk,
    pub index: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionThinkingBlock {
    Thinking {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thinking: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
    RedactedThinking {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionToolCallChunk>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_blocks: Option<Vec<ChatCompletionThinkingBlock>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionStreamingChoice {
    pub index: u64,
    pub delta: ChatCompletionDelta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub created: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub object: String,
    pub choices: Vec<ChatCompletionStreamingChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatCompletionsUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<Map<String, Value>>,
}
