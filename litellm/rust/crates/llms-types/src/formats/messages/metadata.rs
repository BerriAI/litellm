use serde_json::{Map, Value};

use crate::json_schema::JsonSchema;
use crate::recognized::Recognized;

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct MessagesMetadata {
    pub user_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct OutputFormat {
    #[serde(rename = "type")]
    pub format_type: OutputFormatType,
    pub schema: JsonSchema,
    pub strict: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormatType {
    JsonSchema,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct MessagesCompaction {
    #[serde(rename = "type")]
    pub compaction_type: CompactionType,
    pub instructions: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum CompactionType {
    Summarize,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct MessagesContainer {
    pub id: Option<String>,
    pub expires_at: Option<String>,
    pub skills: Option<Vec<ContainerSkill>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ContainerSkill {
    #[serde(rename = "type")]
    pub skill_type: SkillType,
    pub skill_id: String,
    pub version: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum SkillType {
    Anthropic,
    Custom,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct McpServer {
    #[serde(rename = "type")]
    pub server_type: McpServerType,
    pub url: String,
    pub name: String,
    pub authorization_token: Option<String>,
    pub tool_configuration: Option<McpToolConfiguration>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum McpServerType {
    Url,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct McpToolConfiguration {
    pub allowed_tools: Option<Vec<String>>,
    pub enabled: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct StopDetails {
    #[serde(rename = "type")]
    pub detail_type: Recognized<StopDetailsType>,
    pub category: Option<String>,
    pub explanation: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum StopDetailsType {
    Refusal,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ContextManagementResponse {
    pub applied_edits: Option<Vec<AppliedEdit>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct AppliedEdit {
    #[serde(rename = "type")]
    pub edit_type: Option<String>,
    pub cleared_input_tokens: Option<u64>,
    pub cleared_tool_uses: Option<u64>,
    pub cleared_thinking_turns: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct Safeguard {
    #[serde(rename = "type")]
    pub safeguard_type: String,
    pub classifier_context: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::Display,
    strum::EnumString,
    strum::AsRefStr,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(from = "String", into = "String"))]
pub enum MessageRole {
    #[strum(serialize = "user")]
    User,
    #[strum(serialize = "assistant")]
    Assistant,
    #[strum(serialize = "system")]
    System,
    #[strum(default, transparent)]
    Other(String),
}

impl MessageRole {
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

impl From<String> for MessageRole {
    fn from(value: String) -> Self {
        value.parse().unwrap_or_else(|never| match never {})
    }
}

impl From<MessageRole> for String {
    fn from(value: MessageRole) -> Self {
        value.to_string()
    }
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::Display,
    strum::EnumString,
    strum::AsRefStr,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(from = "String", into = "String"))]
pub enum MessageType {
    #[strum(serialize = "message")]
    Message,
    #[strum(default, transparent)]
    Other(String),
}

impl MessageType {
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

impl From<String> for MessageType {
    fn from(value: String) -> Self {
        value.parse().unwrap_or_else(|never| match never {})
    }
}

impl From<MessageType> for String {
    fn from(value: MessageType) -> Self {
        value.to_string()
    }
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::Display,
    strum::EnumString,
    strum::AsRefStr,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(from = "String", into = "String"))]
pub enum StopReason {
    #[strum(serialize = "end_turn")]
    EndTurn,
    #[strum(serialize = "max_tokens")]
    MaxTokens,
    #[strum(serialize = "stop_sequence")]
    StopSequence,
    #[strum(serialize = "tool_use")]
    ToolUse,
    #[strum(serialize = "refusal")]
    Refusal,
    #[strum(serialize = "compaction")]
    Compaction,
    #[strum(serialize = "pause_turn")]
    PauseTurn,
    #[strum(serialize = "model_context_window_exceeded")]
    ModelContextWindowExceeded,
    #[strum(default, transparent)]
    Other(String),
}

impl StopReason {
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

impl From<String> for StopReason {
    fn from(value: String) -> Self {
        value.parse().unwrap_or_else(|never| match never {})
    }
}

impl From<StopReason> for String {
    fn from(value: StopReason) -> Self {
        value.to_string()
    }
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum ContainerReference {
    Id(String),
    Parameters(Box<MessagesContainer>),
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct MessagesDiagnosticsParam {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "::serde_with::rust::double_option"
    )]
    #[cfg_attr(feature = "schema", schemars(with = "Option<String>"))]
    pub previous_message_id: Option<Option<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct MessagesDiagnostics {
    pub cache_miss_reason: Option<Recognized<CacheMissReason>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheMissReason {
    ModelChanged(CacheMissedTokens),
    SystemChanged(CacheMissedTokens),
    ToolsChanged(CacheMissedTokens),
    MessagesChanged(CacheMissedTokens),
    PreviousMessageNotFound {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Unavailable {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct CacheMissedTokens {
    pub cache_missed_input_tokens: u64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {

    use crate::formats::messages::CacheMissReason;

    use crate::formats::messages::ContainerReference;

    use crate::formats::messages::ContextManagementResponse;

    use crate::formats::messages::McpServer;

    use crate::formats::messages::{
        MessageRole, MessageType, MessagesCompaction, MessagesContainer,
    };

    use crate::formats::messages::{
        MessagesDiagnostics, MessagesDiagnosticsParam, MessagesMetadata,
    };

    use crate::formats::messages::OutputFormat;

    use crate::formats::messages::Safeguard;

    use crate::formats::messages::{SkillType, StopDetails, StopDetailsType, StopReason};

    use crate::json_schema::JsonSchema;

    use crate::{recognized::Recognized, test_support::*};

    use rstest::rstest;

    use serde_json::{Value, json};

    #[rstest]
    fn metadata_contracts_round_trip() {
        let metadata = round_trip::<MessagesMetadata>(json!({"user_id":"user_1","future":true}));
        assert_eq!(metadata.user_id.as_deref(), Some("user_1"));
        assert_eq!(metadata.extra.get("future"), Some(&json!(true)));
        let output_format = round_trip::<OutputFormat>(json!({
            "type":"json_schema",
            "schema":{"type":"object","properties":{"name":{"type":"string"}}},
            "strict":true
        }));
        assert_eq!(output_format.strict, Some(true));
        assert!(output_format.extra.is_empty());
        let JsonSchema::Object(schema) = &output_format.schema else {
            panic!("expected output object schema");
        };
        assert!(schema.properties.as_ref().unwrap().contains_key("name"));
        let compaction =
            round_trip::<MessagesCompaction>(json!({"type":"summarize","instructions":"briefly"}));
        assert_eq!(compaction.instructions.as_deref(), Some("briefly"));
        assert!(compaction.extra.is_empty());
        let container = round_trip::<MessagesContainer>(json!({
            "id":"container_1",
            "expires_at":"2026-01-01T00:00:00Z",
            "skills":[{"type":"custom","skill_id":"skill_1","version":"1"}]
        }));
        assert_eq!(container.id.as_deref(), Some("container_1"));
        assert!(container.extra.is_empty());
        let [skill] = container.skills.as_ref().unwrap().as_slice() else {
            panic!("expected container skill");
        };
        assert_eq!(skill.skill_type, SkillType::Custom);
        assert_eq!(skill.skill_id, "skill_1");
        assert_eq!(skill.version.as_deref(), Some("1"));
        assert!(skill.extra.is_empty());
        let reference = round_trip::<ContainerReference>(json!({"id":"container_1"}));
        let ContainerReference::Parameters(parameters) = reference else {
            panic!("expected container parameters");
        };
        assert_eq!(parameters.id.as_deref(), container.id.as_deref());
        assert!(parameters.skills.is_none());
        round_trip::<ContainerReference>(
            json!({"id":"container_1","skills":[{"type":"anthropic","skill_id":"pptx"}]}),
        );
        let server = round_trip::<McpServer>(json!({
            "type":"url",
            "url":"https://example.test/mcp",
            "name":"search",
            "authorization_token":"token",
            "tool_configuration":{"allowed_tools":["search"],"enabled":true}
        }));
        assert_eq!(server.url, "https://example.test/mcp");
        assert_eq!(server.name, "search");
        assert_eq!(server.authorization_token.as_deref(), Some("token"));
        assert!(server.extra.is_empty());
        let configuration = server.tool_configuration.as_ref().unwrap();
        assert_eq!(configuration.enabled, Some(true));
        assert_eq!(
            configuration.allowed_tools.as_deref(),
            Some([String::from("search")].as_slice())
        );
        assert!(configuration.extra.is_empty());
        let context = round_trip::<ContextManagementResponse>(json!({
            "applied_edits":[
                {"type":"clear_tool_uses_20250919","cleared_input_tokens":7,"cleared_tool_uses":2},
                {"type":"clear_thinking_20251015","cleared_input_tokens":3,"cleared_thinking_turns":1}
            ]
        }));
        let [tool_uses, thinking] = context.applied_edits.as_ref().unwrap().as_slice() else {
            panic!("expected two applied edits");
        };
        assert_eq!(
            tool_uses.edit_type.as_deref(),
            Some("clear_tool_uses_20250919")
        );
        assert_eq!(
            (tool_uses.cleared_input_tokens, tool_uses.cleared_tool_uses),
            (Some(7), Some(2))
        );
        assert!(tool_uses.cleared_thinking_turns.is_none());
        assert_eq!(
            (
                thinking.cleared_input_tokens,
                thinking.cleared_thinking_turns
            ),
            (Some(3), Some(1))
        );
        assert!(tool_uses.extra.is_empty() && thinking.extra.is_empty());
        let safeguard = round_trip::<Safeguard>(
            json!({"type":"classifier","classifier_context":{"source":"test"}}),
        );
        assert_eq!(safeguard.safeguard_type, "classifier");
        assert_eq!(
            safeguard.classifier_context.as_ref().unwrap().get("source"),
            Some(&json!("test"))
        );
        assert!(safeguard.extra.is_empty());
    }

    #[rstest]
    fn stop_details_expose_refusal_and_keep_unknown_types() {
        let refusal = round_trip::<StopDetails>(
            json!({"type":"refusal","category":"cyber","explanation":"blocked"}),
        );
        assert_eq!(
            refusal.detail_type,
            Recognized::Known(StopDetailsType::Refusal)
        );
        assert_eq!(refusal.category.as_deref(), Some("cyber"));
        assert_eq!(refusal.explanation.as_deref(), Some("blocked"));
        assert!(refusal.extra.is_empty());
        let safeguard =
            round_trip::<StopDetails>(json!({"type":"safeguard","safeguard_types":["classifier"]}));
        assert_eq!(
            safeguard.detail_type,
            Recognized::Unrecognized(json!("safeguard"))
        );
        assert_eq!(safeguard.extra["safeguard_types"], json!(["classifier"]));
    }

    #[rstest]
    fn container_rejects_skill_without_id() {
        assert!(
            serde_json::from_value::<MessagesContainer>(
                json!({"skills":[{"type":"custom","version":"1"}]})
            )
            .is_err()
        );
    }

    #[rstest]
    #[case::without_url(json!({"type":"url","name":"search"}))]
    #[case::without_name(json!({"type":"url","url":"https://example.test/mcp"}))]
    fn mcp_server_rejects_missing_required_fields(#[case] wire: Value) {
        assert!(serde_json::from_value::<McpServer>(wire).is_err());
    }

    #[rstest]
    fn output_format_rejects_missing_schema() {
        assert!(serde_json::from_value::<OutputFormat>(json!({"type":"json_schema"})).is_err());
    }

    #[rstest]
    #[case::user(json!("user"))]
    #[case::assistant(json!("assistant"))]
    #[case::system(json!("system"))]
    #[case::future(json!("future_role"))]
    fn message_roles_round_trip(#[case] wire: Value) {
        round_trip::<MessageRole>(wire);
    }

    #[rstest]
    #[case::message(json!("message"))]
    #[case::future(json!("future_type"))]
    fn message_types_round_trip(#[case] wire: Value) {
        round_trip::<MessageType>(wire);
    }

    #[rstest]
    #[case::end_turn(json!("end_turn"))]
    #[case::refusal(json!("refusal"))]
    #[case::compaction(json!("compaction"))]
    #[case::future(json!("future_reason"))]
    fn stop_reasons_round_trip(#[case] wire: Value) {
        round_trip::<StopReason>(wire);
    }

    #[rstest]
    #[case::model(json!({"type":"model_changed","cache_missed_input_tokens":12}), Some(12))]
    #[case::system(json!({"type":"system_changed","cache_missed_input_tokens":3}), Some(3))]
    #[case::tools(json!({"type":"tools_changed","cache_missed_input_tokens":0}), Some(0))]
    #[case::messages(json!({"type":"messages_changed","cache_missed_input_tokens":9}), Some(9))]
    #[case::not_found(json!({"type":"previous_message_not_found"}), None)]
    #[case::unavailable(json!({"type":"unavailable"}), None)]
    fn diagnostics_expose_cache_miss_reason(#[case] reason: Value, #[case] missed: Option<u64>) {
        let diagnostics = round_trip::<MessagesDiagnostics>(json!({"cache_miss_reason":reason}));
        let Some(Recognized::Known(reason)) = &diagnostics.cache_miss_reason else {
            panic!("expected known cache miss reason");
        };
        let actual = match reason {
            CacheMissReason::ModelChanged(tokens)
            | CacheMissReason::SystemChanged(tokens)
            | CacheMissReason::ToolsChanged(tokens)
            | CacheMissReason::MessagesChanged(tokens) => {
                assert!(tokens.extra.is_empty());
                Some(tokens.cache_missed_input_tokens)
            }
            CacheMissReason::PreviousMessageNotFound { extra }
            | CacheMissReason::Unavailable { extra } => {
                assert!(extra.is_empty());
                None
            }
        };
        assert_eq!(actual, missed);
        assert!(diagnostics.extra.is_empty());
    }

    #[rstest]
    #[case::model(json!({"type":"model_changed","cache_missed_input_tokens":12}), "model_changed")]
    #[case::system(json!({"type":"system_changed","cache_missed_input_tokens":12}), "system_changed")]
    #[case::tools(json!({"type":"tools_changed","cache_missed_input_tokens":12}), "tools_changed")]
    #[case::messages(json!({"type":"messages_changed","cache_missed_input_tokens":12}), "messages_changed")]
    fn cache_miss_reason_variants_keep_their_tags(#[case] reason: Value, #[case] tag: &str) {
        let parsed: CacheMissReason = serde_json::from_value(reason).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap()["type"], json!(tag));
    }

    #[rstest]
    #[case::pending(json!({"cache_miss_reason":null}))]
    #[case::future_reason(json!({"cache_miss_reason":{"type":"future_changed","cache_missed_input_tokens":1}}))]
    #[case::missing_tokens(json!({"cache_miss_reason":{"type":"model_changed"}}))]
    fn diagnostics_keep_pending_and_unmodeled_reasons(#[case] wire: Value) {
        let diagnostics: MessagesDiagnostics = serde_json::from_value(wire.clone()).unwrap();
        match &diagnostics.cache_miss_reason {
            None => assert_eq!(serde_json::to_value(&diagnostics).unwrap(), json!({})),
            Some(Recognized::Unrecognized(kept)) => {
                assert_eq!(kept, &wire["cache_miss_reason"]);
                assert_eq!(serde_json::to_value(&diagnostics).unwrap(), wire);
            }
            Some(Recognized::Known(reason)) => panic!("unexpected known reason {reason:?}"),
        }
    }

    #[rstest]
    #[case::previous(json!({"previous_message_id":"msg_1"}), Some(Some("msg_1")))]
    #[case::first_turn(json!({"previous_message_id":null}), Some(None))]
    #[case::absent(json!({}), None)]
    fn diagnostics_param_distinguishes_null_from_absent_previous_message(
        #[case] wire: Value,
        #[case] expected: Option<Option<&str>>,
    ) {
        let param = round_trip::<MessagesDiagnosticsParam>(wire);
        assert_eq!(
            param.previous_message_id.as_ref().map(Option::as_deref),
            expected
        );
        assert!(param.extra.is_empty());
    }

    #[rstest]
    fn diagnostics_param_rejects_non_string_previous_message() {
        assert!(
            serde_json::from_value::<MessagesDiagnosticsParam>(json!({"previous_message_id":7}))
                .is_err()
        );
    }

    #[rstest]
    fn cache_miss_reason_rejects_non_numeric_tokens() {
        assert!(
            serde_json::from_value::<CacheMissReason>(
                json!({"type":"model_changed","cache_missed_input_tokens":"many"})
            )
            .is_err()
        );
    }
}
