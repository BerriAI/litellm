use litellm_auth_types::{AwsKeys, Error, Sources, VertexKeys, VertexKeysPartial, VertexParams};
use litellm_layered::{Env, merge};
use rstest::rstest;

fn vertex_params(entries: &[(&str, &str)]) -> VertexParams {
    serde_json::from_value(
        entries
            .iter()
            .map(|(name, value)| (name.to_string(), serde_json::Value::from(*value)))
            .collect::<serde_json::Map<_, _>>()
            .into(),
    )
    .unwrap()
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
#[case::values_are_kept_untrimmed(&[("vertex_location", " eu ")], &[], Some(" eu "))]
#[case::nothing_set(&[], &[], None)]
fn params_are_stacked_above_the_environment(
    #[case] params: &[(&str, &str)],
    #[case] env: &[(&str, &str)],
    #[case] expected: Option<&str>,
) {
    let merged = merge([
        vertex_params(params).keys(),
        VertexKeysPartial::from_env(&Env::of(env)).unwrap(),
    ]);
    assert_eq!(merged.location.as_deref(), expected);
}

#[rstest]
#[case::without_a_litellm_setting(
    AwsKeys::REGION,
    "Missing AWS region - pass aws_region_name or set AWS_REGION_NAME or AWS_REGION"
)]
#[case::with_a_litellm_setting(
    VertexKeys::LOCATION,
    "Missing AWS location - pass vertex_location or vertex_ai_location, set litellm.vertex_location, or set VERTEXAI_LOCATION or VERTEX_LOCATION"
)]
fn a_missing_param_names_every_source_the_group_reads(
    #[case] sources: &'static Sources,
    #[case] expected: &str,
) {
    assert_eq!(Error::missing("AWS", sources).to_string(), expected);
}
