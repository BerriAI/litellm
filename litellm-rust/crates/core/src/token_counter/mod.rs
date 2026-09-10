//! Input token counting for a request body, mirroring `litellm.token_counter`
//! for the shapes it can count exactly. Everything else is declined so the host
//! keeps its own counter as the reference.

mod byte_level;
mod python_json;
mod tools;
pub mod types;
mod unicode_classes;

use serde::Serialize;
use thiserror::Error as ThisError;

use byte_level::ByteLevelCounter;

use crate::constants::{
    NAMED_TOOL_CHOICE_TOKENS, REPLY_PRIMING_TOKENS, TOKENS_PER_MESSAGE, TOKENS_PER_NAME,
    TOOL_CHOICE_NONE_TOKENS, TOOL_DEFINITIONS_TOKENS, TOOLS_WITH_SYSTEM_MESSAGE_DISCOUNT,
};
use tools::format_function_definitions;
use types::{
    ContentBlock, ContentItem, CountableRequest, Message, MessageContent, TextValue, ToolChoice,
};

#[derive(Debug, ThisError, PartialEq, Eq)]
pub enum TokenCountError {
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    /// The body is outside the shape this counter mirrors exactly. Hosts with a
    /// reference counter treat this as "fall back", not "fail".
    #[error("unsupported by the rust token counter: {0}")]
    Unsupported(String),
    #[error("tokenization failed: {0}")]
    Encode(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct InputTokenCount {
    pub model: Option<String>,
    pub input_tokens: usize,
}

/// A loaded HuggingFace tokenizer plus the message accounting Python applies on
/// top of it. Encoding is CPU-bound and synchronous; hosts run it off their
/// event loop.
pub struct TokenCounter {
    tokenizer: tokenizers::Tokenizer,
    byte_level: Option<ByteLevelCounter>,
}

impl TokenCounter {
    /// Load a HuggingFace `tokenizer.json` document. The host reads the file.
    pub fn from_json(tokenizer_json: &str) -> Result<Self, TokenCountError> {
        let tokenizer = tokenizer_json
            .parse::<tokenizers::Tokenizer>()
            .map_err(|error| TokenCountError::Load(error.to_string()))?;
        let byte_level = ByteLevelCounter::detect(&tokenizer);
        Ok(Self {
            tokenizer,
            byte_level,
        })
    }

    pub fn count_text(&self, text: &str) -> Result<usize, TokenCountError> {
        if let Some(count) = self
            .byte_level
            .as_ref()
            .and_then(|counter| counter.count(&self.tokenizer, text))
        {
            return Ok(count);
        }
        self.tokenizer
            .encode_fast(text, true)
            .map(|encoding| encoding.len())
            .map_err(|error| TokenCountError::Encode(error.to_string()))
    }

    /// Mirrors the host's key precedence: `messages`, then `prompt`, then
    /// `input`, then `query` plus `documents`.
    pub fn count_request(
        &self,
        request: &CountableRequest,
    ) -> Result<InputTokenCount, TokenCountError> {
        let input_tokens = if let Some(messages) = &request.messages {
            self.count_messages(request, messages)?
        } else if let Some(prompt) = &request.prompt {
            self.count_text_value(prompt)?
        } else if let Some(input) = &request.input {
            self.count_text_value(input)?
        } else if request.query.is_some() || request.documents.is_some() {
            self.count_optional_text_value(request.query.as_ref())?
                + self.count_optional_text_value(request.documents.as_ref())?
        } else {
            return Err(TokenCountError::Unsupported(
                "request has no countable input".to_string(),
            ));
        };
        Ok(InputTokenCount {
            model: request.model.clone(),
            input_tokens,
        })
    }

    fn count_messages(
        &self,
        request: &CountableRequest,
        messages: &[Message],
    ) -> Result<usize, TokenCountError> {
        let message_tokens = messages
            .iter()
            .map(|message| self.count_message(message))
            .sum::<Result<usize, _>>()?;
        let includes_system_message = messages
            .iter()
            .any(|message| message.role.as_deref() == Some("system"));
        let extra_tokens = self.count_extra(
            request.tools.as_deref().unwrap_or_default(),
            request.tool_choice.as_ref(),
            includes_system_message,
        )?;
        Ok(message_tokens + extra_tokens)
    }

    fn count_optional_text_value(
        &self,
        value: Option<&TextValue>,
    ) -> Result<usize, TokenCountError> {
        value.map_or(Ok(0), |value| self.count_text_value(value))
    }

    /// `str()` for scalars, `json.dumps()` for objects, lists flattened, nulls
    /// skipped. Floats are declined because Python's `repr` and Rust's float
    /// formatting disagree on exponents.
    fn count_text_value(&self, value: &TextValue) -> Result<usize, TokenCountError> {
        match value {
            TextValue::Null => Ok(0),
            TextValue::Bool(true) => self.count_text("True"),
            TextValue::Bool(false) => self.count_text("False"),
            TextValue::Integer(number) => self.count_text(&number.to_string()),
            TextValue::Float(_) => Err(TokenCountError::Unsupported(
                "float text values are counted by the python path".to_string(),
            )),
            TextValue::Text(text) => self.count_text(text),
            TextValue::List(items) => items
                .iter()
                .map(|item| self.count_text_value(item))
                .sum::<Result<usize, _>>(),
            TextValue::Object(_) => self.count_text(&python_json::dumps(value)?),
        }
    }

    fn count_message(&self, message: &Message) -> Result<usize, TokenCountError> {
        let role_tokens = match &message.role {
            Some(role) => self.count_text(role)?,
            None => 0,
        };
        let name_tokens = match &message.name {
            Some(name) => self.count_text(name)? + TOKENS_PER_NAME,
            None => 0,
        };
        let content_tokens = match &message.content {
            Some(MessageContent::Text(text)) => self.count_text(text)?,
            Some(MessageContent::Blocks(items)) => items
                .iter()
                .map(|item| self.count_content_item(item))
                .sum::<Result<usize, _>>()?,
            None => 0,
        };
        Ok(TOKENS_PER_MESSAGE + role_tokens + name_tokens + content_tokens)
    }

    fn count_content_item(&self, item: &ContentItem) -> Result<usize, TokenCountError> {
        match item {
            ContentItem::Text(text) => self.count_text(text),
            ContentItem::Block(ContentBlock::Text { text }) => self.count_text(text),
            ContentItem::Block(ContentBlock::Thinking { thinking }) => {
                if thinking.is_empty() {
                    return Ok(0);
                }
                self.count_text(thinking)
            }
            ContentItem::Block(ContentBlock::ToolReference { tool_name }) => {
                match tool_name.as_deref().filter(|name| !name.is_empty()) {
                    Some(name) => self.count_text(name),
                    None => Ok(0),
                }
            }
            ContentItem::Block(ContentBlock::Unsupported) => Err(TokenCountError::Unsupported(
                "content block type is counted by the python path".to_string(),
            )),
        }
    }

    fn count_extra(
        &self,
        tools: &[types::ToolDefinition],
        tool_choice: Option<&ToolChoice>,
        includes_system_message: bool,
    ) -> Result<usize, TokenCountError> {
        let tool_tokens = if tools.is_empty() {
            0
        } else {
            let definitions = self.count_text(&format_function_definitions(tools)?)?;
            let discount = if includes_system_message {
                TOOLS_WITH_SYSTEM_MESSAGE_DISCOUNT
            } else {
                0
            };
            definitions + TOOL_DEFINITIONS_TOKENS - discount
        };
        let choice_tokens = match tool_choice {
            Some(ToolChoice::Mode(mode)) if mode == "none" => TOOL_CHOICE_NONE_TOKENS,
            Some(ToolChoice::Mode(_)) | None => 0,
            Some(ToolChoice::Named(named)) => {
                NAMED_TOOL_CHOICE_TOKENS + self.count_text(&named.function.name)?
            }
        };
        Ok(REPLY_PRIMING_TOKENS + tool_tokens + choice_tokens)
    }
}

#[cfg(test)]
mod tests;
