use litellm_auth_types::AwsParams;
use rstest::rstest;
use serde_json::{Map, Value};

#[rstest]
fn every_named_param_roundtrips_without_loss() {
    let filled: Map<String, Value> = AwsParams::fields()
        .map(|name| (name.to_string(), Value::from(format!("value-of-{name}"))))
        .collect();
    let typed = AwsParams::from_optional_params(&filled);
    let serialized = serde_json::to_value(&typed).unwrap();
    assert_eq!(filled.len(), AwsParams::fields().count());
    assert_eq!(serialized, Value::Object(filled.clone()));
    assert_eq!(
        serde_json::from_value::<AwsParams>(Value::Object(filled)).unwrap(),
        typed
    );
}
