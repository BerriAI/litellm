use serde_json::{Map, Value};

use crate::recognized::Recognized;
use crate::serde_compat::deserialize_present;

use super::CacheControl;
use crate::json_schema::JsonSchema;

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ToolDefinition {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub name: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub description: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub input_schema: Option<Recognized<JsonSchema>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub strict: Option<Recognized<bool>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cache_control: Option<Recognized<CacheControl>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub defer_loading: Option<Recognized<bool>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub allowed_callers: Option<Recognized<Vec<String>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub input_examples: Option<Recognized<Vec<Map<String, Value>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub eager_input_streaming: Option<Recognized<bool>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub display_width_px: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub display_height_px: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub display_number: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub max_uses: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub max_tokens: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub allowed_domains: Option<Recognized<Vec<String>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub blocked_domains: Option<Recognized<Vec<String>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub citations: Option<Recognized<super::CitationsConfig>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub user_location: Option<Recognized<WebSearchUserLocation>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub model: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub caching: Option<Recognized<CacheControl>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct WebSearchUserLocation {
    #[serde(rename = "type")]
    pub location_type: UserLocationType,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub city: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub country: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub region: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub timezone: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum UserLocationType {
    Approximate,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct ToolChoice {
    #[serde(rename = "type")]
    pub choice_type: ToolChoiceType,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub name: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub disable_parallel_tool_use: Option<Recognized<bool>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceType {
    Auto,
    Any,
    Tool,
    None,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::formats::messages::{BuiltinMessagesTool, CustomTool, MessagesTool};
    use crate::serde_compat::Nullable;

    #[rstest]
    #[case::custom(None, json!({"name":"lookup","input_schema":false,"strict":false,"input_examples":[{"cache_control":{"ttl":"1h"},"content":[{"type":"thinking","signature":null}]}],"cache_control":{"type":"ephemeral","ttl":null,"scope":"global"}}))]
    #[case::web_search(Some(BuiltinMessagesTool::WebSearch as fn(ToolDefinition) -> BuiltinMessagesTool), json!({"type":"web_search_20250305","name":"lookup","input_schema":false,"strict":false,"input_examples":[{"cache_control":{"ttl":"1h"},"content":[{"type":"thinking","signature":null}]}],"cache_control":{"type":"ephemeral","ttl":null,"scope":"global"}}))]
    #[case::web_search_v2(Some(BuiltinMessagesTool::WebSearch20260209 as fn(ToolDefinition) -> BuiltinMessagesTool), json!({"type":"web_search_20260209","name":"lookup","input_schema":false,"strict":false,"input_examples":[{"cache_control":{"ttl":"1h"},"content":[{"type":"thinking","signature":null}]}],"cache_control":{"type":"ephemeral","ttl":null,"scope":"global"}}))]
    fn tool_schemas_examples_and_cache_control_preserve_wire_values(
        #[case] builtin: Option<fn(ToolDefinition) -> BuiltinMessagesTool>,
        #[case] wire: Value,
    ) {
        let definition = ToolDefinition {
            name: Some(Recognized::Known("lookup".into())),
            input_schema: Some(Recognized::Known(JsonSchema::Boolean(false))),
            strict: Some(Recognized::Known(false)),
            input_examples: Some(Recognized::Known(vec![Map::from_iter([
                ("cache_control".into(), json!({"ttl":"1h"})),
                (
                    "content".into(),
                    json!([{"type":"thinking","signature":null}]),
                ),
            ])])),
            cache_control: Some(Recognized::Known(CacheControl {
                cache_type: Some(Nullable::Value("ephemeral".into())),
                ttl: Some(Nullable::Null),
                scope: Some(Nullable::Value("global".into())),
                extra: Map::new(),
            })),
            ..Default::default()
        };
        let tool = match builtin {
            Some(constructor) => MessagesTool::Builtin(constructor(definition)),
            None => MessagesTool::Custom(CustomTool::try_from(definition).unwrap()),
        };
        assert_eq!(serde_json::to_value(&tool).unwrap(), wire);
        assert_eq!(serde_json::from_value::<MessagesTool>(wire).unwrap(), tool);
    }

    #[rstest]
    #[case::auto(ToolChoiceType::Auto, "auto")]
    #[case::any(ToolChoiceType::Any, "any")]
    #[case::tool(ToolChoiceType::Tool, "tool")]
    #[case::none(ToolChoiceType::None, "none")]
    fn tool_choices_keep_explicit_parallel_control(
        #[case] choice_type: ToolChoiceType,
        #[case] wire_type: &str,
    ) {
        let choice = ToolChoice {
            choice_type,
            name: Some(Recognized::Known("lookup".into())),
            disable_parallel_tool_use: Some(Recognized::Known(false)),
            extra: Map::new(),
        };
        let wire = json!({"type":wire_type,"name":"lookup","disable_parallel_tool_use":false});
        assert_eq!(serde_json::to_value(&choice).unwrap(), wire);
        assert_eq!(serde_json::from_value::<ToolChoice>(wire).unwrap(), choice);
    }
}
