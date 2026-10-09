use litellm_auth_aws::AwsParams;
use litellm_auth_gcp::VertexParams;
use serde::Deserialize;

/// The typed subset of Python's `GenericLiteLLMParams` that provider configs read.
///
/// One flattened group per credential family, so a host projects exactly [`Self::fields`]
/// out of a caller's kwargs and a config reaches for `litellm_params.aws` or
/// `litellm_params.vertex`, never a map.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct LitellmParams {
    #[serde(flatten)]
    pub aws: AwsParams,
    #[serde(flatten)]
    pub vertex: VertexParams,
}

impl LitellmParams {
    /// The wire names a host projects into this type; everything else in the kwargs is not
    /// a litellm param the configs read.
    pub fn fields() -> impl Iterator<Item = &'static str> {
        AwsParams::FIELDS.into_iter().chain(VertexParams::FIELDS)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    #[rstest]
    #[case::aws_group(
        json!({"aws_region_name": "eu-west-1", "aws_access_key_id": "AKIA"}),
        LitellmParams {
            aws: AwsParams {
                aws_region_name: Some("eu-west-1".into()),
                aws_access_key_id: Some("AKIA".into()),
                ..AwsParams::default()
            },
            ..LitellmParams::default()
        },
    )]
    #[case::vertex_group(
        json!({"vertex_project": "p", "vertex_ai_location": "eu", "vertex_credentials": {"type": "service_account"}}),
        LitellmParams {
            vertex: VertexParams {
                vertex_project: Some("p".into()),
                vertex_ai_location: Some("eu".into()),
                vertex_credentials: Some(r#"{"type":"service_account"}"#.into()),
                ..VertexParams::default()
            },
            ..LitellmParams::default()
        },
    )]
    #[case::both_groups(
        json!({"aws_region_name": "eu-west-1", "vertex_location": "us-east5"}),
        LitellmParams {
            aws: AwsParams {
                aws_region_name: Some("eu-west-1".into()),
                ..AwsParams::default()
            },
            vertex: VertexParams {
                vertex_location: Some("us-east5".into()),
                ..VertexParams::default()
            },
        },
    )]
    #[case::explicit_null_is_absent(
        json!({"aws_region_name": null, "vertex_project": null}),
        LitellmParams::default(),
    )]
    #[case::keys_outside_the_groups_are_ignored(
        json!({"messages": [], "max_tokens": 5, "azure_ad_token": "t"}),
        LitellmParams::default(),
    )]
    fn deserializes_the_credential_groups_from_caller_kwargs(
        #[case] kwargs: Value,
        #[case] expected: LitellmParams,
    ) {
        assert_eq!(
            serde_json::from_value::<LitellmParams>(kwargs).unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::aws(json!({"aws_region_name": 7}))]
    #[case::vertex(json!({"vertex_project": ["p"]}))]
    #[case::vertex_credentials(json!({"vertex_credentials": 7}))]
    fn a_param_of_the_wrong_type_is_rejected(#[case] kwargs: Value) {
        assert!(serde_json::from_value::<LitellmParams>(kwargs).is_err());
    }

    #[test]
    fn fields_fill_every_group() {
        let filled: Value = LitellmParams::fields()
            .map(|name| (name.to_string(), Value::from(name)))
            .collect::<serde_json::Map<_, _>>()
            .into();
        let typed: LitellmParams = serde_json::from_value(filled).unwrap();
        let nulls = |group: Value| {
            group
                .as_object()
                .unwrap()
                .values()
                .filter(|value| value.is_null())
                .count()
        };
        assert_eq!(nulls(serde_json::to_value(&typed.aws).unwrap()), 0);
        assert_eq!(nulls(serde_json::to_value(&typed.vertex).unwrap()), 0);
    }
}
