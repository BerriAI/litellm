use litellm_auth_types::AwsParams;
use rstest::rstest;
use serde_json::{Map, Value};

#[rstest]
fn fields_name_every_param_once_and_in_declaration_order() {
    let filled: Map<String, Value> = AwsParams::fields()
        .map(|name| (name.to_string(), Value::from(format!("value-of-{name}"))))
        .collect();
    let typed = AwsParams::from_optional_params(&filled);
    let serialized = serde_json::to_value(&typed).unwrap();
    assert_eq!(
        serialized
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        AwsParams::fields().collect::<Vec<_>>()
    );
    assert_eq!(serialized, Value::Object(filled.clone()));
    assert_eq!(
        serde_json::from_value::<AwsParams>(Value::Object(filled)).unwrap(),
        typed
    );
}
