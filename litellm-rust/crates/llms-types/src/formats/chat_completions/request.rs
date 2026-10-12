use serde_json::{Map, Value};
use strum::IntoStaticStr;

/// Reasoning effort level accepted or applied by the model.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq, IntoStaticStr, strum::EnumString, strum::VariantArray)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    #[strum(serialize = "none")]
    None,
    #[strum(serialize = "minimal")]
    Minimal,
    #[strum(serialize = "low")]
    Low,
    #[strum(serialize = "medium")]
    Medium,
    #[strum(serialize = "high")]
    High,
    #[strum(serialize = "xhigh")]
    Xhigh,
    #[strum(serialize = "max")]
    Max,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<Value>),
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatMessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use strum::VariantArray;

    #[rstest]
    fn reasoning_effort_names_match_the_wire_and_parse_back(
        #[values(
            ReasoningEffort::None,
            ReasoningEffort::Minimal,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::Xhigh,
            ReasoningEffort::Max
        )]
        effort: ReasoningEffort,
    ) {
        assert_eq!(
            serde_json::to_value(effort).unwrap(),
            Value::String(<&'static str>::from(effort).to_string())
        );
        assert_eq!(<&'static str>::from(effort).parse(), Ok(effort));
        assert!(ReasoningEffort::VARIANTS.contains(&effort));
    }

    #[rstest]
    #[case::unknown("ultra")]
    #[case::uppercase("HIGH")]
    #[case::empty("")]
    fn reasoning_effort_parse_rejects(#[case] value: &str) {
        assert!(value.parse::<ReasoningEffort>().is_err());
    }
}
