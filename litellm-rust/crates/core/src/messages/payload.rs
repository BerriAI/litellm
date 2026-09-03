use super::types::{
    AnthropicMessage, AnthropicMessagesRequest, CacheControl, ContentBlock, MessageContent,
    SystemPrompt,
};
use crate::Error;
use crate::http_utils::body::JsonPayload;

fn metadata(value: impl serde::Serialize) -> Result<JsonPayload, Error> {
    serde_json::to_value(value)
        .map(JsonPayload::from)
        .map_err(|_| Error::InvalidRequest("invalid messages metadata".into()))
}

impl AnthropicMessagesRequest {
    pub(crate) fn into_payload(self) -> Result<JsonPayload, Error> {
        let mut fields = self.extra;
        fields.insert("model".into(), self.model.into());
        fields.insert(
            "messages".into(),
            JsonPayload::Array(
                self.messages
                    .into_iter()
                    .map(AnthropicMessage::into_payload)
                    .collect::<Result<_, _>>()?,
            ),
        );
        if let Some(system) = self.system {
            let content = match system {
                SystemPrompt::Text(text) => MessageContent::Text(text),
                SystemPrompt::Blocks(blocks) => MessageContent::Blocks(blocks),
            };
            fields.insert("system".into(), content.into_payload()?);
        }
        macro_rules! optional {
            ($($field:ident => $convert:expr),* $(,)?) => { $(
                if let Some(value) = self.$field { fields.insert(stringify!($field).into(), $convert(value)?); }
            )* };
        }
        optional! {
            max_tokens => metadata,
            stop_sequences => metadata,
            stream => metadata,
            temperature => metadata,
            top_p => metadata,
            top_k => metadata,
            service_tier => metadata,
            speed => metadata,
            inference_geo => metadata,
            metadata => Ok::<_, Error>,
            tool_choice => Ok::<_, Error>,
            thinking => Ok::<_, Error>,
            container => Ok::<_, Error>,
            context_management => Ok::<_, Error>,
            output_format => Ok::<_, Error>,
            output_config => Ok::<_, Error>,
            tools => |value| Ok::<_, Error>(JsonPayload::Array(value)),
            mcp_servers => |value| Ok::<_, Error>(JsonPayload::Array(value)),
        }
        Ok(JsonPayload::Object(fields))
    }
}

impl AnthropicMessage {
    fn into_payload(self) -> Result<JsonPayload, Error> {
        let mut fields = self.extra;
        fields.insert("role".into(), self.role.into());
        fields.insert("content".into(), self.content.into_payload()?);
        Ok(JsonPayload::Object(fields))
    }
}

impl MessageContent {
    fn into_payload(self) -> Result<JsonPayload, Error> {
        match self {
            Self::Text(text) => Ok(JsonPayload::String(text)),
            Self::Blocks(blocks) => blocks
                .into_iter()
                .map(ContentBlock::into_payload)
                .collect::<Result<_, _>>()
                .map(JsonPayload::Array),
        }
    }
}

impl ContentBlock {
    fn into_payload(self) -> Result<JsonPayload, Error> {
        let mut fields = self.extra;
        if let Some(cache) = self.cache_control {
            fields.insert("cache_control".into(), cache.into_payload()?);
        }
        Ok(JsonPayload::Object(fields))
    }
}

impl CacheControl {
    fn into_payload(self) -> Result<JsonPayload, Error> {
        let mut fields = self.extra;
        for (key, value) in [
            ("type", self.cache_type),
            ("ttl", self.ttl),
            ("scope", self.scope),
        ] {
            if let Some(value) = value {
                fields.insert(key.into(), value.into());
            }
        }
        Ok(JsonPayload::Object(fields))
    }
}
