use serde_json::{Map, Value};

use crate::recognized::Recognized;
use crate::serde_compat::deserialize_present;

use crate::serde_compat::Nullable;

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentSource {
    Base64 {
        media_type: String,
        data: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Url {
        url: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    File {
        file_id: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Text {
        media_type: String,
        data: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Content {
        content: Recognized<BlockContent>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum BlockContent {
    Text(String),
    Blocks(Vec<Recognized<ContentBlock>>),
    SearchError(WebSearchResultError),
    Block(Box<ContentBlock>),
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCaller {
    Direct {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    #[serde(rename = "code_execution_20250825")]
    CodeExecution {
        tool_id: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct CitationsConfig {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub enabled: Option<Recognized<bool>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct PageCitation {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cited_text: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub document_index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub document_title: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub start_page_number: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub end_page_number: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct CharCitation {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cited_text: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub document_index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub document_title: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub start_char_index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub end_char_index: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct WebSearchCitation {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cited_text: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub url: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub title: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub encrypted_index: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Citation {
    PageLocation(PageCitation),
    CharLocation(CharCitation),
    WebSearchResultLocation(WebSearchCitation),
    ContentBlockLocation(ContentBlockCitation),
    SearchResultLocation(SearchResultCitation),
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum Citations {
    Config(CitationsConfig),
    Results(Vec<Recognized<Citation>>),
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct WebSearchResultError {
    #[serde(rename = "type")]
    pub error_type: WebSearchResultErrorType,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub error_code: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchResultErrorType {
    WebSearchToolResultError,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct PromptCacheBreakpoint {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub mode: Option<Recognized<PromptCacheMode>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheMode {
    Explicit,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ContentBlockCitation {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cited_text: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub document_index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub document_title: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub start_block_index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub end_block_index: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct SearchResultCitation {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cited_text: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub search_result_index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub title: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub source: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub start_block_index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub end_block_index: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::Display,
    strum::EnumString,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(from = "String", into = "String"))]
#[strum(serialize_all = "snake_case")]
pub enum ContentBlockType {
    Text,
    Image,
    Document,
    ContainerUpload,
    ToolReference,
    SearchResult,
    WebSearchResult,
    WebFetchResult,
    WebFetchToolResult,
    WebFetchToolResultError,
    CodeExecutionToolResult,
    CodeExecutionToolResultError,
    CodeExecutionResult,
    CodeExecutionOutput,
    EncryptedCodeExecutionResult,
    BashCodeExecutionToolResult,
    BashCodeExecutionToolResultError,
    BashCodeExecutionResult,
    BashCodeExecutionOutput,
    TextEditorCodeExecutionToolResult,
    TextEditorCodeExecutionToolResultError,
    TextEditorCodeExecutionViewResult,
    TextEditorCodeExecutionCreateResult,
    TextEditorCodeExecutionStrReplaceResult,
    ToolSearchToolResult,
    ToolSearchToolResultError,
    ToolSearchToolSearchResult,
    McpToolUse,
    McpToolResult,
    AdvisorResult,
    AdvisorRedactedResult,
    WebSearchToolResultError,
    Thinking,
    RedactedThinking,
    ToolUse,
    ToolAddition,
    ToolRemoval,
    ServerToolUse,
    ToolResult,
    Compaction,
    AdvisorToolResult,
    WebSearchToolResult,
    #[strum(default, transparent)]
    Other(String),
}

impl From<String> for ContentBlockType {
    fn from(value: String) -> Self {
        value.parse().unwrap_or_else(|never| match never {})
    }
}

impl From<ContentBlockType> for String {
    fn from(value: ContentBlockType) -> Self {
        value.to_string()
    }
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ContentBlock {
    #[serde(rename = "type", default, deserialize_with = "deserialize_present")]
    pub block_type: Option<Nullable<ContentBlockType>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub tool_use_id: Option<Nullable<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cache_control: Option<Nullable<CacheControl>>,
    #[serde(flatten)]
    pub payload: ContentBlockPayload,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ContentBlockPayload {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub tool: Option<Recognized<Box<ContentBlock>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub text: Option<Nullable<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub thinking: Option<Nullable<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub signature: Option<Nullable<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub data: Option<Nullable<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub id: Option<Nullable<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub name: Option<Nullable<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub input: Option<Recognized<Map<String, Value>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub content: Option<Recognized<BlockContent>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub provider_specific_fields: Option<Recognized<Map<String, Value>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub source: Option<Recognized<ContentSource>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub citations: Option<Recognized<Citations>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub caller: Option<Recognized<ToolCaller>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub is_error: Option<Recognized<bool>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub file_id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub title: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub context: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub tool_name: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub url: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub page_age: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub encrypted_content: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub snippet: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub prompt_cache_breakpoint: Option<Recognized<PromptCacheBreakpoint>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub stdout: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub stderr: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub return_code: Option<Recognized<i64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub encrypted_stdout: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub error_code: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub error_message: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub retrieved_at: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub server_name: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub tool_references: Option<Recognized<Vec<Recognized<super::ContentBlock>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub file_type: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub num_lines: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub start_line: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub total_lines: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub is_file_update: Option<Recognized<bool>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub lines: Option<Recognized<Vec<String>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub new_lines: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub new_start: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub old_lines: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub old_start: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl std::ops::Deref for ContentBlock {
    type Target = ContentBlockPayload;

    fn deref(&self) -> &Self::Target {
        &self.payload
    }
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            block_type: Some(Nullable::Value(ContentBlockType::Text)),
            payload: ContentBlockPayload {
                text: Some(Nullable::Value(text.into())),
                ..ContentBlockPayload::default()
            },
            ..Self::default()
        }
    }

    pub fn is_type(&self, block_type: ContentBlockType) -> bool {
        self.block_type.as_ref().and_then(Nullable::value) == Some(&block_type)
    }
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct CacheControl {
    #[serde(rename = "type", default, deserialize_with = "deserialize_present")]
    pub cache_type: Option<Nullable<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub ttl: Option<Nullable<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub scope: Option<Nullable<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::tool_addition("tool_addition", ContentBlockType::ToolAddition)]
    #[case::tool_removal("tool_removal", ContentBlockType::ToolRemoval)]
    #[case::unknown("future_change", ContentBlockType::Other("future_change".into()))]
    #[case::case_sensitive("Tool_Addition", ContentBlockType::Other("Tool_Addition".into()))]
    #[case::empty("", ContentBlockType::Other(String::new()))]
    fn native_messages_tool_changes_discriminator(
        #[case] wire: &str,
        #[case] expected: ContentBlockType,
    ) {
        assert_eq!(
            serde_json::from_value::<ContentBlockType>(json!(wire)).unwrap(),
            expected
        );
        assert_eq!(serde_json::to_value(&expected).unwrap(), json!(wire));
        let tool = json!({"type":"tool_reference","name":"mcp__test__ping","input":{"type":"tool_removal","signature":null}});
        let block = ContentBlock {
            block_type: Some(Nullable::Value(expected)),
            payload: ContentBlockPayload {
                tool: Some(Recognized::Known(Box::new(ContentBlock {
                    block_type: Some(Nullable::Value(ContentBlockType::ToolReference)),
                    payload: ContentBlockPayload {
                        name: Some(Nullable::Value("mcp__test__ping".into())),
                        input: Some(Recognized::Known(Map::from_iter([
                            ("type".into(), json!("tool_removal")),
                            ("signature".into(), Value::Null),
                        ]))),
                        ..Default::default()
                    },
                    ..Default::default()
                }))),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_value(&block).unwrap(),
            json!({"type":wire,"tool":tool})
        );
        assert_eq!(
            serde_json::from_value::<ContentBlock>(json!({"type":wire,"tool":tool})).unwrap(),
            block
        );
    }

    #[rstest]
    #[case::signed_thinking(ContentBlockType::Thinking, Some(Nullable::Value("plan".into())), Some(Nullable::Value("EqQBCkYIAxgCIkA_anthropic_signed".into())), None, json!({"type":"thinking","thinking":"plan","signature":"EqQBCkYIAxgCIkA_anthropic_signed"}))]
    #[case::redacted_thinking(ContentBlockType::RedactedThinking, None, None, Some(Nullable::Value("EmwKAhgBEgy_anthropic_minted".into())), json!({"type":"redacted_thinking","data":"EmwKAhgBEgy_anthropic_minted"}))]
    #[case::null_signature(ContentBlockType::Thinking, Some(Nullable::Value("let me think".into())), Some(Nullable::Null), None, json!({"type":"thinking","thinking":"let me think","signature":null}))]
    #[case::missing_signature(ContentBlockType::Thinking, Some(Nullable::Value("let me think".into())), None, None, json!({"type":"thinking","thinking":"let me think"}))]
    fn anthropic_signed_thinking_payload_is_preserved(
        #[case] block_type: ContentBlockType,
        #[case] thinking: Option<Nullable<String>>,
        #[case] signature: Option<Nullable<String>>,
        #[case] data: Option<Nullable<String>>,
        #[case] wire: Value,
    ) {
        let block = ContentBlock {
            block_type: Some(Nullable::Value(block_type)),
            payload: ContentBlockPayload {
                thinking,
                signature,
                data,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(serde_json::to_value(&block).unwrap(), wire);
        assert_eq!(serde_json::from_value::<ContentBlock>(wire).unwrap(), block);
    }

    #[rstest]
    fn tool_input_and_provider_fields_remain_opaque() {
        let input = Map::from_iter([
            ("cache_control".into(), json!({"ttl":"1h","scope":null})),
            (
                "content".into(),
                json!([{"type":"thinking","signature":"encrypted","provider_specific_fields":{"type":null}}]),
            ),
        ]);
        let provider_fields = Map::from_iter([("signature".into(), json!({"nested":[null,true]}))]);
        let block = ContentBlock {
            block_type: Some(Nullable::Value(ContentBlockType::ToolUse)),
            payload: ContentBlockPayload {
                id: Some(Nullable::Value("toolu_1".into())),
                name: Some(Nullable::Value("lookup".into())),
                input: Some(Recognized::Known(input.clone())),
                provider_specific_fields: Some(Recognized::Known(provider_fields.clone())),
                ..Default::default()
            },
            ..Default::default()
        };
        let wire = json!({"type":"tool_use","id":"toolu_1","name":"lookup","input":input,"provider_specific_fields":provider_fields});
        assert_eq!(serde_json::to_value(&block).unwrap(), wire);
        assert_eq!(serde_json::from_value::<ContentBlock>(wire).unwrap(), block);
    }

    #[rstest]
    #[case::null(json!(null))]
    #[case::scalar(json!(17))]
    #[case::malformed_type(json!({"type":false,"name":"lookup"}))]
    #[case::malformed_name(json!({"type":"tool_reference","name":17}))]
    fn tool_changes_preserve_unrecognized_tool_payloads(#[case] tool: Value) {
        let wire = json!({"type":"tool_addition","tool":tool});
        let block: ContentBlock = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(block.tool, Some(Recognized::Unrecognized(tool)));
        let stream_block: crate::formats::messages::streaming::MessagesContentBlock =
            serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(stream_block.payload, block.payload);
        assert_eq!(serde_json::to_value(stream_block).unwrap(), wire);
        assert_eq!(serde_json::to_value(block).unwrap(), wire);
    }

    #[rstest]
    fn tool_changes_preserve_unknown_nested_discriminators() {
        let wire = json!({"type":"tool_removal","tool":{"type":"future_reference","future":{"nested":[null,true]}}});
        let block: ContentBlock = serde_json::from_value(wire.clone()).unwrap();
        let tool = block.tool.as_ref().and_then(Recognized::known).unwrap();
        assert!(tool.is_type(ContentBlockType::Other("future_reference".into())));
        assert_eq!(
            tool.extra.get("future"),
            Some(&json!({"nested":[null,true]}))
        );
        let stream_block: crate::formats::messages::streaming::MessagesContentBlock =
            serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(stream_block.payload, block.payload);
        assert_eq!(serde_json::to_value(stream_block).unwrap(), wire);
        assert_eq!(serde_json::to_value(block).unwrap(), wire);
    }
}
