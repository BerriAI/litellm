//! Opt-in proxy identity for Bedrock's STS session attribution.

use std::{
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

use fancy_regex::Regex;
use litellm_auth::AwsParams;

pub const REQUIRE_SESSION_ATTRIBUTION: &str = "bedrock_require_session_attribution";
/// Configured attribution fields and the request's string-valued proxy identity.
#[derive(Default)]
pub struct SessionIdentity {
    pub tag_fields: Vec<String>,
    pub name_fields: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub litellm_metadata: HashMap<String, String>,
}

static TAG_TEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\A[\p{L}\p{Z}\p{N}_.:/=+\-@]+\z").expect("constant STS tag pattern")
});
static SESSION_NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\A[a-zA-Z0-9_+=,.@-]{2,64}\z").expect("constant STS session name pattern")
});

const CALLER_SUPPLIED_FIELDS: [&str; 1] = ["user_api_key_end_user_id"];

fn identity<'a>(params: &'a SessionIdentity, field: &str) -> Option<&'a str> {
    if !field.starts_with("user_api_key_") || CALLER_SUPPLIED_FIELDS.contains(&field) {
        return None;
    }
    [&params.metadata, &params.litellm_metadata]
        .into_iter()
        .find_map(|source| {
            source
                .get(field)
                .map(String::as_str)
                .filter(|value| !value.is_empty())
        })
}

pub fn session_identity(
    aws: &AwsParams,
    params: &SessionIdentity,
    env: &dyn Fn(&str) -> Option<String>,
) -> AwsParams {
    if (aws.aws_role_name.is_none() && env("AWS_ROLE_NAME").is_none())
        || aws.aws_web_identity_token.is_some()
        || env("AWS_WEB_IDENTITY_TOKEN").is_some()
    {
        return aws.clone();
    }
    let mut tags = aws.aws_session_tags.clone().unwrap_or_default();
    let mut keys: HashSet<String> = tags.iter().map(|tag| tag.key.to_lowercase()).collect();
    for field in &params.tag_fields {
        let Some(value) = identity(params, field) else {
            continue;
        };
        if tags.len() >= 50
            || keys.contains(&field.to_lowercase())
            || field.chars().count() > 128
            || value.chars().count() > 256
            || !TAG_TEXT.is_match(field).unwrap_or(false)
            || !TAG_TEXT.is_match(value).unwrap_or(false)
        {
            continue;
        }
        keys.insert(field.to_lowercase());
        tags.push(litellm_auth::AwsSessionTag {
            key: field.to_owned(),
            value: value.to_owned(),
        });
    }
    let session_name = aws
        .aws_session_name
        .clone()
        .or_else(|| env("AWS_SESSION_NAME"))
        .or_else(|| {
            params
                .name_fields
                .iter()
                .filter_map(|field| identity(params, field))
                .find(|value| SESSION_NAME.is_match(value).unwrap_or(false))
                .map(str::to_owned)
        });
    AwsParams {
        aws_session_tags: if tags.is_empty() {
            aws.aws_session_tags.clone()
        } else {
            Some(tags)
        },
        aws_session_name: session_name,
        ..aws.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use litellm_auth::AwsSessionTag;
    use rstest::{fixture, rstest};

    #[fixture]
    fn aws() -> AwsParams {
        AwsParams {
            aws_role_name: Some("arn:aws:iam::123456789012:role/invocation".into()),
            ..Default::default()
        }
    }

    #[rstest]
    #[case::unicode("équipe 東京 42".into(), true)]
    #[case::punctuation("agent_.:/=+-@".into(), true)]
    #[case::max_length("x".repeat(256), true)]
    #[case::too_long("x".repeat(257), false)]
    #[case::empty("".into(), false)]
    #[case::invalid("agent!".into(), false)]
    #[case::newline("agent\nname".into(), false)]
    fn derived_tags_keep_only_valid_unmodified_values(
        aws: AwsParams,
        #[case] value: String,
        #[case] valid: bool,
    ) {
        let identity = SessionIdentity {
            tag_fields: vec!["user_api_key_alias".into()],
            metadata: [("user_api_key_alias".into(), value.clone())].into(),
            ..Default::default()
        };
        let result = session_identity(&aws, &identity, &|_| None);
        let expected = valid.then_some(vec![AwsSessionTag {
            key: "user_api_key_alias".into(),
            value,
        }]);
        assert_eq!(result.aws_session_tags, expected);
    }

    #[rstest]
    #[case::first("agent-one".into(), "agent-one")]
    #[case::too_short("a".into(), "agent-two")]
    #[case::too_long("x".repeat(65), "agent-two")]
    #[case::unicode("équipe".into(), "agent-two")]
    #[case::invalid("agent/name".into(), "agent-two")]
    #[case::empty("".into(), "agent-two")]
    fn names_use_the_first_valid_fallback(
        aws: AwsParams,
        #[case] first: String,
        #[case] expected: &str,
    ) {
        let identity = SessionIdentity {
            name_fields: vec!["user_api_key_user_id".into(), "user_api_key_alias".into()],
            litellm_metadata: [
                ("user_api_key_user_id".into(), first),
                ("user_api_key_alias".into(), "agent-two".into()),
            ]
            .into(),
            ..Default::default()
        };
        assert_eq!(
            session_identity(&aws, &identity, &|_| None)
                .aws_session_name
                .as_deref(),
            Some(expected)
        );
    }

    #[rstest]
    #[case::static_conflict(1)]
    #[case::full_budget(50)]
    fn static_tags_win_and_never_exceed_the_budget(mut aws: AwsParams, #[case] count: usize) {
        let static_tags: Vec<_> = (0..count)
            .map(|index| AwsSessionTag {
                key: if index == 0 {
                    "USER_API_KEY_ALIAS".into()
                } else {
                    format!("static-{index}")
                },
                value: "deployment".into(),
            })
            .collect();
        aws.aws_session_tags = Some(static_tags.clone());
        let identity = SessionIdentity {
            tag_fields: vec!["user_api_key_alias".into()],
            metadata: [("user_api_key_alias".into(), "caller".into())].into(),
            ..Default::default()
        };
        assert_eq!(
            session_identity(&aws, &identity, &|_| None).aws_session_tags,
            Some(static_tags)
        );
    }

    #[rstest]
    #[case::explicit(Some("explicit"), Some("environment"), "explicit")]
    #[case::ambient(None, Some("environment"), "environment")]
    #[case::derived(None, None, "agent")]
    fn explicit_and_environment_names_win(
        mut aws: AwsParams,
        #[case] explicit: Option<&str>,
        #[case] ambient: Option<&str>,
        #[case] expected: &str,
    ) {
        aws.aws_session_name = explicit.map(str::to_owned);
        let identity = SessionIdentity {
            name_fields: vec!["user_api_key_alias".into()],
            metadata: [("user_api_key_alias".into(), "agent".into())].into(),
            ..Default::default()
        };
        let resolved = session_identity(&aws, &identity, &|key| {
            (key == "AWS_SESSION_NAME")
                .then_some(ambient)
                .flatten()
                .map(str::to_owned)
        });
        assert_eq!(resolved.aws_session_name.as_deref(), Some(expected));
    }

    #[rstest]
    fn caller_supplied_end_user_id_is_never_used(aws: AwsParams) {
        let identity = SessionIdentity {
            tag_fields: vec!["user_api_key_end_user_id".into()],
            name_fields: vec!["user_api_key_end_user_id".into()],
            metadata: [("user_api_key_end_user_id".into(), "caller-chosen".into())].into(),
            ..Default::default()
        };
        let resolved = session_identity(&aws, &identity, &|_| None);
        assert_eq!(
            (resolved.aws_session_tags, resolved.aws_session_name),
            (None, None)
        );
    }

    #[rstest]
    fn absent_role_does_not_add_attribution() {
        let identity = SessionIdentity {
            tag_fields: vec!["user_api_key_alias".into()],
            name_fields: vec!["user_api_key_alias".into()],
            metadata: [("user_api_key_alias".into(), "agent".into())].into(),
            ..Default::default()
        };
        let resolved = session_identity(&AwsParams::default(), &identity, &|_| None);
        assert_eq!(
            (resolved.aws_session_tags, resolved.aws_session_name),
            (None, None)
        );
    }
}
