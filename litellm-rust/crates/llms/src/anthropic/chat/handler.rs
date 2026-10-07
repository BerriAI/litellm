use std::collections::HashMap;

use litellm_types::{
    llms::openai::{ChatCompletionThinkingBlock, ChatCompletionToolCallChunk},
    utils::{ChatCompletionChunk, ChatCompletionsUsage},
};
use serde_json::Value;

use crate::{
    anthropic::experimental_pass_through::messages::streaming_iterator::{
        AnthropicContentBlock, AnthropicContentBlockDelta, AnthropicMessagesStreamEvent,
        AnthropicStreamUsage,
    },
    base_llm::{base_model_iterator::StreamTransformer, chat::transformation::Error},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnthropicJsonChunkType {
    ValidJson,
    AccumulatedJson,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnthropicContentBlockType {
    Text,
    ToolUse,
    ServerToolUse,
    Thinking,
    RedactedThinking,
    Compaction,
    ToolResult(String),
    Other(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnthropicContentBlockDeltaEvent {
    pub index: u64,
    pub delta: AnthropicContentBlockDelta,
}

pub struct AnthropicChatCompletionsStreamTransformer {
    pub content_blocks: Vec<AnthropicContentBlockDeltaEvent>,
    pub tool_index: i64,
    pub json_mode: bool,
    pub speed: Option<String>,
    pub tool_name_reverse_map: HashMap<String, String>,
    pub response_id: String,
    pub served_model: Option<String>,
    pub is_response_format_tool: bool,
    pub converted_response_format_tool: bool,
    pub accumulated_json: String,
    pub chunk_type: AnthropicJsonChunkType,
    pub current_content_block_type: Option<AnthropicContentBlockType>,
    pub web_search_results: Vec<Value>,
    pub web_search_calls: HashMap<String, Value>,
    pub compaction_blocks: Vec<Value>,
    pub reasoning_content_chunks: Vec<String>,
    pub server_tool_inputs: HashMap<String, Value>,
    pub tool_results: Vec<Value>,
    pub current_server_tool_id: Option<String>,
    pub container_id: Option<String>,
}

impl AnthropicChatCompletionsStreamTransformer {
    pub fn new(
        _json_mode: bool,
        _speed: Option<String>,
        _tool_name_reverse_map: HashMap<String, String>,
    ) -> Self {
        todo!()
    }

    pub fn check_empty_tool_call_args(&self) -> bool {
        todo!()
    }

    pub fn handle_usage(&mut self, _usage: AnthropicStreamUsage) -> ChatCompletionsUsage {
        todo!()
    }

    pub fn handle_content_block_delta(
        &mut self,
        _index: u64,
        _delta: AnthropicContentBlockDelta,
    ) -> (
        String,
        Option<ChatCompletionToolCallChunk>,
        Vec<ChatCompletionThinkingBlock>,
        Option<Value>,
        Option<String>,
    ) {
        todo!()
    }

    pub fn handle_content_block_start(
        &mut self,
        _index: u64,
        _content_block: AnthropicContentBlock,
    ) -> Result<ChatCompletionChunk, Error> {
        todo!()
    }

    pub fn handle_json_mode_chunk(
        &mut self,
        _text: String,
        _tool_use: Option<ChatCompletionToolCallChunk>,
    ) -> (String, Option<ChatCompletionToolCallChunk>) {
        todo!()
    }

    pub fn handle_accumulated_json_chunk(
        &mut self,
        _data: &str,
        _is_final: bool,
    ) -> Result<Option<ChatCompletionChunk>, Error> {
        todo!()
    }

    pub fn handle_redacted_thinking_content(
        &mut self,
        _content_block: &AnthropicContentBlock,
    ) -> Vec<ChatCompletionThinkingBlock> {
        todo!()
    }

    pub fn web_search_call_snapshot(&self) -> HashMap<String, Value> {
        todo!()
    }

    pub fn complete_web_search_call(&mut self, _result: Value) {
        todo!()
    }

    pub fn build_code_interpreter_results(&self) -> Vec<Value> {
        todo!()
    }

    pub fn handle_message_delta(
        &mut self,
        _event: AnthropicMessagesStreamEvent,
    ) -> (Option<String>, Option<ChatCompletionsUsage>, Option<Value>) {
        todo!()
    }

    pub fn chunk_parser(
        &mut self,
        _event: AnthropicMessagesStreamEvent,
    ) -> Result<ChatCompletionChunk, Error> {
        todo!()
    }
}

impl StreamTransformer for AnthropicChatCompletionsStreamTransformer {
    type Input = AnthropicMessagesStreamEvent;
    type Output = ChatCompletionChunk;
    type Error = Error;

    fn transform(&mut self, _input: Self::Input) -> Result<Vec<Self::Output>, Self::Error> {
        todo!()
    }

    fn finish(&mut self) -> Result<Vec<Self::Output>, Self::Error> {
        todo!()
    }
}
