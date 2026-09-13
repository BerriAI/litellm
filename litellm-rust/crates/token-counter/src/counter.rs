use serde::Serialize;

use crate::Error;
use crate::byte_level::ByteLevelCounter;
use crate::python_json;
use crate::scanner::{SplitPattern, TiktokenCounter};
use crate::tools::format_function_definitions;
use crate::types::{
    ContentBlock, ContentItem, CountableRequest, Message, MessageContent, TextValue, ToolChoice,
    ToolDefinition,
};

const TOKENS_PER_MESSAGE: usize = 3;
const TOKENS_PER_NAME: usize = 1;
const REPLY_PRIMING_TOKENS: usize = 3;
const TOOL_DEFINITIONS_TOKENS: usize = 9;
const TOOLS_WITH_SYSTEM_MESSAGE_DISCOUNT: usize = 4;
const TOOL_CHOICE_NONE_TOKENS: usize = 1;
const NAMED_TOOL_CHOICE_TOKENS: usize = 7;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct InputTokenCount {
    pub model: Option<String>,
    pub input_tokens: usize,
}

enum Encoder {
    HuggingFace {
        tokenizer: Box<tokenizers::Tokenizer>,
        byte_level: Option<ByteLevelCounter>,
    },
    Tiktoken(TiktokenCounter),
}

/// A loaded tokenizer plus the message accounting Python applies on top of
/// it. Encoding is CPU-bound and synchronous; hosts run it off their event
/// loop.
pub struct TokenCounter {
    encoder: Encoder,
}

impl TokenCounter {
    /// Load a HuggingFace `tokenizer.json` document. The host reads the file.
    pub fn from_json(tokenizer_json: &str) -> Result<Self, Error> {
        let tokenizer = tokenizer_json
            .parse::<tokenizers::Tokenizer>()
            .map_err(Error::Load)?;
        let byte_level = ByteLevelCounter::detect(&tokenizer);
        Ok(Self {
            encoder: Encoder::HuggingFace {
                tokenizer: Box::new(tokenizer),
                byte_level,
            },
        })
    }

    /// Load tiktoken's `cl100k_base` rank file (`base64(token) rank` lines).
    /// The host reads the file.
    pub fn from_cl100k_ranks(rank_file: &str) -> Result<Self, Error> {
        Self::from_tiktoken_ranks(SplitPattern::Cl100k, rank_file)
    }

    /// Load tiktoken's `o200k_base` rank file (`base64(token) rank` lines).
    /// The host reads the file.
    pub fn from_o200k_ranks(rank_file: &str) -> Result<Self, Error> {
        Self::from_tiktoken_ranks(SplitPattern::O200k, rank_file)
    }

    fn from_tiktoken_ranks(split: SplitPattern, rank_file: &str) -> Result<Self, Error> {
        Ok(Self {
            encoder: Encoder::Tiktoken(TiktokenCounter::from_ranks(split, rank_file)?),
        })
    }

    pub fn count_text(&self, text: &str) -> Result<usize, Error> {
        match &self.encoder {
            Encoder::Tiktoken(counter) => Ok(counter.count(text)),
            Encoder::HuggingFace {
                tokenizer,
                byte_level,
            } => {
                if let Some(count) = byte_level
                    .as_ref()
                    .and_then(|counter| counter.count(tokenizer, text))
                {
                    return Ok(count);
                }
                tokenizer
                    .encode_fast(text, true)
                    .map(|encoding| encoding.len())
                    .map_err(Error::Encode)
            }
        }
    }

    /// Mirrors the host's key precedence: `messages`, then `prompt`, then
    /// `input`, then `query` plus `documents`.
    pub fn count_request(&self, request: &CountableRequest) -> Result<InputTokenCount, Error> {
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
            return Err(Error::MissingInput);
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
    ) -> Result<usize, Error> {
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

    fn count_optional_text_value(&self, value: Option<&TextValue>) -> Result<usize, Error> {
        value.map_or(Ok(0), |value| self.count_text_value(value))
    }

    /// `str()` for scalars, `json.dumps()` for objects, lists flattened, nulls
    /// skipped. Floats are declined because Python's `repr` and Rust's float
    /// formatting disagree on exponents.
    fn count_text_value(&self, value: &TextValue) -> Result<usize, Error> {
        match value {
            TextValue::Null => Ok(0),
            TextValue::Bool(true) => self.count_text("True"),
            TextValue::Bool(false) => self.count_text("False"),
            TextValue::Number(number) => match (number.as_i64(), number.as_u64()) {
                (Some(number), _) => self.count_text(itoa::Buffer::new().format(number)),
                (_, Some(number)) => self.count_text(itoa::Buffer::new().format(number)),
                _ => Err(Error::FloatText),
            },
            TextValue::Text(text) => self.count_text(text),
            TextValue::List(items) => items
                .iter()
                .map(|item| self.count_text_value(item))
                .sum::<Result<usize, _>>(),
            TextValue::Object(_) => self.count_text(&python_json::dumps(value)?),
        }
    }

    fn count_message(&self, message: &Message) -> Result<usize, Error> {
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

    fn count_content_item(&self, item: &ContentItem) -> Result<usize, Error> {
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
            ContentItem::Block(ContentBlock::Unsupported) => Err(Error::ContentBlock),
        }
    }

    fn count_extra(
        &self,
        tools: &[ToolDefinition],
        tool_choice: Option<&ToolChoice>,
        includes_system_message: bool,
    ) -> Result<usize, Error> {
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
