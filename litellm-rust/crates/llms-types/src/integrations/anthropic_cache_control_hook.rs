use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

use crate::{
    formats::messages::CacheControl,
    message_role::MessageRole,
    recognized::Recognized,
    serde_compat::{PYTHON_DEFAULT_INT_MAX_STR_DIGITS, ParsedInteger, parse_python_integer},
};

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "location", rename = "message")]
pub struct CacheControlMessageInjectionPoint {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "serde_with::rust::double_option"
    )]
    control: Option<Option<CacheControl>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "serde_with::rust::double_option"
    )]
    index: Option<Option<CacheControlIndex>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "serde_with::rust::double_option"
    )]
    role: Option<Option<Recognized<MessageRole>>>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

impl CacheControlMessageInjectionPoint {
    pub fn new(
        index: Option<i64>,
        role: Option<MessageRole>,
        control: Option<CacheControl>,
    ) -> Self {
        Self {
            control: control.map(Some),
            index: index.map(|index| Some(CacheControlIndex::Number(index.into()))),
            role: role.map(|role| Some(Recognized::Known(role))),
            extra: Map::new(),
        }
    }

    pub fn parsed_index(&self) -> ParsedInteger {
        match self.index.as_ref().and_then(Option::as_ref) {
            Some(CacheControlIndex::Number(number)) => number
                .as_i64()
                .map(ParsedInteger::Value)
                .unwrap_or(ParsedInteger::OutOfRange),
            Some(CacheControlIndex::String(index)) => {
                parse_python_integer(index, PYTHON_DEFAULT_INT_MAX_STR_DIGITS)
            }
            Some(CacheControlIndex::Boolean(value)) => ParsedInteger::Value(i64::from(*value)),
            Some(CacheControlIndex::Other(_)) | None => ParsedInteger::Invalid,
        }
    }

    pub fn role(&self) -> Option<&MessageRole> {
        self.role
            .as_ref()
            .and_then(Option::as_ref)
            .and_then(Recognized::known)
    }

    pub fn control(&self) -> Option<&CacheControl> {
        self.control.as_ref().and_then(Option::as_ref)
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
enum CacheControlIndex {
    Number(serde_json::Number),
    String(String),
    Boolean(bool),
    Other(Value),
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "location", rename = "tool_config")]
pub struct CacheControlToolConfigInjectionPoint {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "serde_with::rust::double_option"
    )]
    control: Option<Option<CacheControl>>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum CacheControlInjectionPoint {
    Message(CacheControlMessageInjectionPoint),
    ToolConfig(CacheControlToolConfigInjectionPoint),
    Unknown(Value),
}

impl<'de> Deserialize<'de> for CacheControlInjectionPoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value.get("location").and_then(Value::as_str) {
            Some("message") => serde_json::from_value(value).map(Self::Message),
            Some("tool_config") => serde_json::from_value(value).map(Self::ToolConfig),
            _ => return Ok(Self::Unknown(value)),
        }
        .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::{CacheControlInjectionPoint, CacheControlMessageInjectionPoint};
    use crate::message_role::MessageRole;
    use crate::serde_compat::ParsedInteger;

    #[rstest]
    #[case::negative_index_string(json!({"location": "message", "index": "-1"}), ParsedInteger::Value(-1))]
    #[case::padded_index_string(json!({"location": "message", "index": " \t0\n"}), ParsedInteger::Value(0))]
    #[case::out_of_range_index(json!({"location": "message", "index": 99}), ParsedInteger::Value(99))]
    #[case::oversized_string(json!({"location": "message", "index": "9223372036854775808", "role": "user"}), ParsedInteger::OutOfRange)]
    #[case::oversized_number(json!({"location": "message", "index": u64::MAX, "role": "user"}), ParsedInteger::OutOfRange)]
    #[case::underscores(json!({"location": "message", "index": "0_0"}), ParsedInteger::Value(0))]
    #[case::unicode_digits(json!({"location": "message", "index": "٠"}), ParsedInteger::Value(0))]
    #[case::signed_and_padded(json!({"location": "message", "index": " +0_1 "}), ParsedInteger::Value(1))]
    #[case::negative_limit(json!({"location": "message", "index": "-9223372036854775808"}), ParsedInteger::Value(i64::MIN))]
    #[case::negative_overflow(json!({"location": "message", "index": "-9223372036854775809", "role": "user"}), ParsedInteger::OutOfRange)]
    #[case::boolean_index(json!({"location": "message", "index": true}), ParsedInteger::Value(1))]
    #[case::mixed_digit_scripts(json!({"location": "message", "index": "٠_１"}), ParsedInteger::Value(1))]
    #[case::mathematical_digit(json!({"location": "message", "index": "𝟙"}), ParsedInteger::Value(1))]
    #[case::double_underscore(json!({"location": "message", "index": "0__1", "role": "user"}), ParsedInteger::Invalid)]
    #[case::leading_underscore(json!({"location": "message", "index": "_1", "role": "user"}), ParsedInteger::Invalid)]
    #[case::trailing_underscore(json!({"location": "message", "index": "1_", "role": "user"}), ParsedInteger::Invalid)]
    #[case::sign_without_digits(json!({"location": "message", "index": "+", "role": "user"}), ParsedInteger::Invalid)]
    #[case::embedded_sign(json!({"location": "message", "index": "1+0", "role": "user"}), ParsedInteger::Invalid)]
    #[case::non_decimal_digit(json!({"location": "message", "index": "²", "role": "user"}), ParsedInteger::Invalid)]
    #[case::role_only(json!({"location": "message", "role": "user"}), ParsedInteger::Invalid)]
    #[case::invalid_index_falls_back_to_role(
        json!({"location": "message", "index": "latest", "role": "assistant"}),
        ParsedInteger::Invalid
    )]
    #[case::index_precedes_system_role(
        json!({"location": "message", "index": 0, "role": "system"}),
        ParsedInteger::Value(0)
    )]
    #[case::missing_target(json!({"location": "message"}), ParsedInteger::Invalid)]
    fn message_index_deserialization(#[case] input: Value, #[case] expected: ParsedInteger) {
        let point: CacheControlInjectionPoint = serde_json::from_value(input).unwrap();

        assert!(matches!(
            point,
            CacheControlInjectionPoint::Message(message) if message.parsed_index() == expected
        ));
    }

    #[rstest]
    #[case::missing_control(json!({"location": "message", "index": 0}))]
    #[case::message_null_control(json!({"location": "message", "index": 0, "control": null}))]
    #[case::tool_config_without_control(json!({"location": "tool_config"}))]
    #[case::tool_config_null_control(json!({"location": "tool_config", "control": null}))]
    fn known_points_preserve_optional_control(#[case] input: Value) {
        let point: CacheControlInjectionPoint = serde_json::from_value(input.clone()).unwrap();

        assert_eq!(serde_json::to_value(point).unwrap(), input);
    }

    #[rstest]
    #[case::message(json!({"location": "message", "index": 0, "control": "ephemeral"}))]
    #[case::tool_config(json!({"location": "tool_config", "control": 5}))]
    fn malformed_controls_are_rejected(#[case] input: Value) {
        assert!(serde_json::from_value::<CacheControlInjectionPoint>(input).is_err());
    }

    #[rstest]
    fn unknown_location_round_trips_unchanged() {
        let input = json!({"location": "future", "index": 7, "custom": {"value": true}});
        let point: CacheControlInjectionPoint = serde_json::from_value(input.clone()).unwrap();

        assert!(matches!(&point, CacheControlInjectionPoint::Unknown(_)));
        assert_eq!(serde_json::to_value(point).unwrap(), input);
    }

    #[rstest]
    #[case::role_preserved_when_index_wins(json!({"location": "message", "index": 0, "role": "assistant"}))]
    #[case::unparsed_index_preserved_when_role_wins(json!({"location": "message", "index": "latest", "role": "assistant"}))]
    #[case::message(json!({"location": "message", "index": " 0 ", "role": "user", "control": null, "custom": "keep"}))]
    #[case::tool_config(json!({"location": "tool_config", "control": null, "custom": "keep"}))]
    #[case::null_control_fields(json!({"location": "message", "index": null, "role": null, "control": {"type": null, "ttl": null, "scope": null}}))]
    #[case::unknown_target_types(json!({"location": "message", "index": [1], "role": {"custom": true}}))]
    #[case::unicode_index(json!({"location": "message", "index": " +٠_１ ", "role": "assistant"}))]
    fn known_points_preserve_wire_fields(#[case] input: Value) {
        let point: CacheControlInjectionPoint = serde_json::from_value(input.clone()).unwrap();

        assert_eq!(serde_json::to_value(point).unwrap(), input);
    }

    #[rstest]
    fn constructed_points_preserve_both_index_and_role() {
        let point =
            CacheControlMessageInjectionPoint::new(Some(0), Some(MessageRole::System), None);

        assert_eq!(
            serde_json::to_value(point).unwrap(),
            json!({"location": "message", "index": 0, "role": "system"})
        );
    }
}
