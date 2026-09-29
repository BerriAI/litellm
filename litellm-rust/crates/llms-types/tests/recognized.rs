use litellm_llms_types::recognized::{Recognized, deserialize_present};
use rstest::rstest;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
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
