use litellm_llms_types::{headers::ProviderSpecificHeader, recognized::Recognized};
use rstest::rstest;
use serde_json::json;

#[rstest]
fn header_strings_are_typed_while_legacy_non_string_values_are_preserved() {
    let wire = json!({"custom_llm_provider":"provider","extra_headers":{"valid":"value","number":17,"null":null,"list":["one","two"]}});
    let parsed: ProviderSpecificHeader = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(
        parsed.extra_headers["valid"],
        Recognized::Known("value".into())
    );
    assert_eq!(
        parsed.extra_headers["number"],
        Recognized::Unrecognized(json!(17))
    );
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}
