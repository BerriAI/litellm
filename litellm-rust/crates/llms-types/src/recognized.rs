use serde::{Deserialize, Deserializer};
use serde_json::Value;

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum Recognized<T> {
    Known(T),
    Unrecognized(Value),
}

pub fn deserialize_present<'de, T, D>(deserializer: D) -> Result<Option<Recognized<T>>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    Recognized::deserialize(deserializer).map(Some)
}

impl<T> Recognized<T> {
    pub fn known(&self) -> Option<&T> {
        match self {
            Self::Known(value) => Some(value),
            Self::Unrecognized(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::known(json!(7), Recognized::Known(7))]
    #[case::wrong_type(json!("7"), Recognized::Unrecognized(json!("7")))]
    #[case::out_of_range(json!(-1), Recognized::Unrecognized(json!(-1)))]
    #[case::object(json!({"a": 1}), Recognized::Unrecognized(json!({"a": 1})))]
    fn value_is_known_only_when_it_parses_as_the_type(
        #[case] value: Value,
        #[case] expected: Recognized<u64>,
    ) {
        assert_eq!(
            serde_json::from_value::<Recognized<u64>>(value.clone()).unwrap(),
            expected
        );
        assert_eq!(serde_json::to_value(expected).unwrap(), value);
    }

    #[derive(Debug, PartialEq, serde::Serialize, Deserialize)]
    struct Payload {
        #[serde(
            default,
            deserialize_with = "deserialize_present",
            skip_serializing_if = "Option::is_none"
        )]
        value: Option<Recognized<String>>,
    }

    #[rstest]
    #[case::missing(json!({}), None)]
    #[case::null(json!({"value": null}), Some(Recognized::Unrecognized(Value::Null)))]
    #[case::known(json!({"value": "text"}), Some(Recognized::Known("text".into())))]
    #[case::unknown(json!({"value": 3}), Some(Recognized::Unrecognized(json!(3))))]
    fn optional_recognized_values_preserve_presence(
        #[case] wire: Value,
        #[case] expected: Option<Recognized<String>>,
    ) {
        let payload: Payload = serde_json::from_value(wire.clone()).unwrap();

        assert_eq!(payload.value, expected);
        assert_eq!(serde_json::to_value(payload).unwrap(), wire);
    }
}
