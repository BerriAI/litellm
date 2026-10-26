use std::collections::BTreeSet;

use litellm_auth_types::{AwsKeys, AwsParams};
use rstest::rstest;
use serde_json::{Map, Value};

#[rstest]
fn fields_name_every_param_once() {
    let filled: Map<String, Value> = AwsKeys::LITELLM_PARAMS
        .iter()
        .map(|name| (name.to_string(), Value::from(format!("value-of-{name}"))))
        .collect();
    let typed = AwsParams::from_optional_params(&filled);
    let serialized = serde_json::to_value(&typed).unwrap();
    assert_eq!(serialized, Value::Object(filled.clone()));
    assert_eq!(
        serde_json::from_value::<AwsParams>(Value::Object(filled)).unwrap(),
        typed
    );
    assert_eq!(
        AwsKeys::LITELLM_PARAMS
            .iter()
            .collect::<BTreeSet<_>>()
            .len(),
        AwsKeys::LITELLM_PARAMS.len()
    );
}
