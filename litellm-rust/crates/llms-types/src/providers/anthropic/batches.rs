use serde_json::Value;

use crate::formats::messages::MessagesResponse;

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default, Eq)]
pub struct AnthropicBatchRequestCounts {
    #[serde(default)]
    pub processing: u64,
    #[serde(default)]
    pub succeeded: u64,
    #[serde(default)]
    pub errored: u64,
    #[serde(default)]
    pub canceled: u64,
    #[serde(default)]
    pub expired: u64,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq)]
pub struct AnthropicMessageBatch {
    #[serde(default)]
    pub id: String,
    #[serde(default = "default_processing_status")]
    pub processing_status: String,
    pub created_at: Option<String>,
    pub ended_at: Option<String>,
    pub expires_at: Option<String>,
    pub cancel_initiated_at: Option<String>,
    pub archived_at: Option<String>,
    #[serde(default)]
    pub request_counts: AnthropicBatchRequestCounts,
}

fn default_processing_status() -> String {
    "in_progress".into()
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct AnthropicBatchResultRecord {
    pub result: AnthropicBatchResult,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicBatchResult {
    Succeeded { message: Box<MessagesResponse> },
    Errored { error: Value },
}
