use litellm_auth_types::{AwsParams, ParamSpec};
use rstest::rstest;

const LOCATION: ParamSpec = ParamSpec {
    setting: "location",
    wire: &["vertex_location", "vertex_ai_location"],
    module_global: Some("vertex_location"),
    env: &["VERTEXAI_LOCATION", "VERTEX_LOCATION"],
};

fn lookup(entries: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
    move |name| {
        entries
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value.to_string())
    }
}

#[rstest]
#[case::current_spelling_first(
    &[("vertex_location", "eu"), ("vertex_ai_location", "us")],
    &[("VERTEXAI_LOCATION", "env")],
    Some("eu")
)]
#[case::legacy_spelling_before_the_environment(
    &[("vertex_ai_location", "us")],
    &[("VERTEXAI_LOCATION", "env")],
    Some("us")
)]
#[case::blank_values_are_skipped(
    &[("vertex_location", "  "), ("vertex_ai_location", "")],
    &[("VERTEXAI_LOCATION", " "), ("VERTEX_LOCATION", "fallback")],
    Some("fallback")
)]
#[case::environment_names_in_order(
    &[],
    &[("VERTEX_LOCATION", "second"), ("VERTEXAI_LOCATION", "first")],
    Some("first")
)]
#[case::nothing_set(&[], &[], None)]
fn resolution_walks_wire_names_then_environment_names(
    #[case] params: &'static [(&'static str, &'static str)],
    #[case] env: &'static [(&'static str, &'static str)],
    #[case] expected: Option<&str>,
) {
    assert_eq!(
        LOCATION.resolve(&lookup(params), &lookup(env)).as_deref(),
        expected
    );
}

#[rstest]
#[case::without_a_module_global(
    &AwsParams::REGION,
    "Missing AWS region - pass aws_region_name or set AWS_REGION_NAME or AWS_REGION"
)]
#[case::with_a_module_global(
    &LOCATION,
    "Missing AWS location - pass vertex_location or vertex_ai_location, set litellm.vertex_location, or set VERTEXAI_LOCATION or VERTEX_LOCATION"
)]
fn a_missing_param_names_every_place_the_spec_reads(
    #[case] spec: &'static ParamSpec,
    #[case] expected: &str,
) {
    let message = spec.missing("AWS").to_string();

    assert_eq!(message, expected);
    assert!(spec.wire.iter().all(|name| message.contains(name)));
    assert!(spec.env.iter().all(|name| message.contains(name)));
    assert_eq!(message.contains("litellm."), spec.module_global.is_some());
}
