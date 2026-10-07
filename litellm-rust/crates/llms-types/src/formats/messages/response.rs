use serde_json::{Map, Value};

use super::{
    ContentBlock, ContextManagementResponse, MessagesContainer, MessagesUsage, StopDetails,
};
use crate::recognized::Recognized;
use crate::serde_compat::{Nullable, deserialize_present};

#[macro_rules_attribute::apply(wire_type)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: super::MessageType,
    pub role: super::MessageRole,
    pub model: String,
    pub content: Vec<Recognized<ContentBlock>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub stop_reason: Option<Nullable<super::StopReason>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub stop_sequence: Option<Nullable<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub usage: Option<Recognized<MessagesUsage>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub container: Option<Recognized<MessagesContainer>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub stop_details: Option<Recognized<StopDetails>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub context_management: Option<Recognized<ContextManagementResponse>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub safeguard_results: Option<Recognized<Vec<Map<String, Value>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
