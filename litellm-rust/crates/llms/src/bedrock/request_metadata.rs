use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{Error, base_llm::auth::Headers};

pub const BEDROCK_REQUEST_METADATA_HEADER: &str = "X-Amzn-Bedrock-Request-Metadata";
pub const BEDROCK_REQUEST_METADATA_MAX_PAIRS: usize = 16;
const IDENTITY_PREFIX: &str = "user_api_key_";
const CLIENT_FIELD: &str = "spend_logs_metadata";

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BedrockMetadataSource {
    pub identity: Vec<(String, String)>,
    pub spend_logs: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BedrockRequestMetadataInput {
    pub allowed_fields: Vec<String>,
    pub sources: Vec<BedrockMetadataSource>,
}

fn valid_text(text: &str, allow_empty: bool) -> bool {
    (allow_empty || !text.is_empty())
        && text.chars().count() <= 256
        && text.chars().all(|c| {
            c.is_ascii_alphanumeric()
                || c.is_whitespace()
                || matches!(
                    c,
                    '\u{1c}'
                        ..='\u{1f}'
                            | ':'
                            | '_'
                            | '@'
                            | '$'
                            | '#'
                            | '='
                            | '/'
                            | '+'
                            | ','
                            | '.'
                            | '-'
                )
        })
}

fn forwardable(key: &str, value: &str) -> bool {
    valid_text(key, false) && valid_text(value, true)
}

pub fn resolve_bedrock_request_metadata(
    input: &BedrockRequestMetadataInput,
) -> Vec<(String, String)> {
    let identity: Vec<_> = input
        .allowed_fields
        .iter()
        .enumerate()
        .filter(|(index, field)| {
            field.starts_with(IDENTITY_PREFIX) && !input.allowed_fields[..*index].contains(field)
        })
        .filter_map(|(_, field)| {
            let value = input.sources.iter().find_map(|source| {
                source
                    .identity
                    .iter()
                    .find(|(key, _)| key == field)
                    .map(|(_, value)| value)
            })?;
            forwardable(field, value).then(|| (field.clone(), value.clone()))
        })
        .take(BEDROCK_REQUEST_METADATA_MAX_PAIRS)
        .collect();
    let candidates: Vec<_> = input
        .sources
        .iter()
        .flat_map(|source| source.spend_logs.iter())
        .filter(|(key, value)| !key.starts_with(IDENTITY_PREFIX) && forwardable(key, value))
        .collect();
    let client = candidates
        .iter()
        .enumerate()
        .filter(|(index, (key, _))| {
            !candidates[..*index]
                .iter()
                .any(|(earlier, _)| earlier == key)
        })
        .take(
            if input
                .allowed_fields
                .iter()
                .any(|field| field == CLIENT_FIELD)
            {
                BEDROCK_REQUEST_METADATA_MAX_PAIRS - identity.len()
            } else {
                0
            },
        )
        .map(|(_, pair)| (*pair).clone());
    identity.into_iter().chain(client).collect()
}

pub fn bedrock_request_metadata_headers(
    headers: Headers,
    input: &BedrockRequestMetadataInput,
) -> Result<Headers, Error> {
    if input.allowed_fields.is_empty() {
        return Ok(headers);
    }
    let resolved = resolve_bedrock_request_metadata(input);
    let value = if resolved.is_empty() {
        None
    } else {
        let object: Map<String, Value> = resolved
            .into_iter()
            .map(|(key, value)| (key, Value::String(value)))
            .collect();
        Some(serde_json::to_string(&object).map_err(|error| {
            Error::InvalidRequest(crate::ErrorDetail::invalid(
                "Bedrock request metadata",
                error,
            ))
        })?)
    };
    Ok(headers
        .into_iter()
        .filter(|(name, _)| !name.eq_ignore_ascii_case(BEDROCK_REQUEST_METADATA_HEADER))
        .chain(value.map(|value| (BEDROCK_REQUEST_METADATA_HEADER.into(), value)))
        .collect())
}

#[cfg(test)]
mod tests {
    use litellm_auth_aws::{SigV4Signer, constants::BEDROCK_SERVICE};
    use litellm_http::outbound::OutboundRequest;
    use rstest::{fixture, rstest};
    use serde_json::json;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use super::*;

    #[fixture]
    fn input() -> BedrockRequestMetadataInput {
        BedrockRequestMetadataInput {
            allowed_fields: [
                "user_api_key_alias",
                "user_api_key_team_alias",
                "user_api_key_user_email",
                CLIENT_FIELD,
            ]
            .map(str::to_string)
            .to_vec(),
            sources: vec![BedrockMetadataSource {
                identity: vec![
                    ("user_api_key_alias".into(), "prod-key".into()),
                    ("user_api_key_team_alias".into(), "platform".into()),
                ],
                spend_logs: vec![("cost_center".into(), "cc-1".into())],
            }],
        }
    }

    #[rstest]
    #[case::declared_order(false, false)]
    #[case::reversed_clients(false, true)]
    #[case::reversed_fields(true, false)]
    #[case::both_reversed(true, true)]
    fn identity_survives_a_full_client_budget(
        mut input: BedrockRequestMetadataInput,
        #[case] reverse_fields: bool,
        #[case] reverse_clients: bool,
    ) {
        if reverse_fields {
            input.allowed_fields.reverse();
        }
        input.sources[0].spend_logs = (0..BEDROCK_REQUEST_METADATA_MAX_PAIRS)
            .map(|index| (format!("client_{index:02}"), "v".into()))
            .collect();
        if reverse_clients {
            input.sources[0].spend_logs.reverse();
        }
        let resolved = resolve_bedrock_request_metadata(&input);
        assert_eq!(resolved.len(), BEDROCK_REQUEST_METADATA_MAX_PAIRS);
        assert!(resolved.contains(&("user_api_key_alias".into(), "prod-key".into())));
        assert!(resolved.contains(&("user_api_key_team_alias".into(), "platform".into())));
        assert_eq!(
            resolved
                .iter()
                .filter(|(key, _)| key.starts_with("client_"))
                .count(),
            BEDROCK_REQUEST_METADATA_MAX_PAIRS - 2
        );
    }

    #[rstest]
    #[case::alias_twice(0, "user_api_key_alias")]
    #[case::alias_after_team(2, "user_api_key_alias")]
    #[case::team_after_clients(4, "user_api_key_team_alias")]
    fn repeated_fields_do_not_consume_slots(
        mut input: BedrockRequestMetadataInput,
        #[case] index: usize,
        #[case] repeated: &str,
    ) {
        input.allowed_fields.insert(index, repeated.to_string());
        input.sources[0].spend_logs = (0..BEDROCK_REQUEST_METADATA_MAX_PAIRS - 1)
            .map(|index| (format!("client_{index:02}"), "v".into()))
            .collect();
        let resolved = resolve_bedrock_request_metadata(&input);
        let expected: Vec<_> = [
            ("user_api_key_alias".into(), "prod-key".into()),
            ("user_api_key_team_alias".into(), "platform".into()),
        ]
        .into_iter()
        .chain(
            (0..BEDROCK_REQUEST_METADATA_MAX_PAIRS - 2)
                .map(|index| (format!("client_{index:02}"), "v".into())),
        )
        .collect();
        assert_eq!(resolved, expected);
    }

    #[rstest]
    #[case::team("user_api_key_team_alias")]
    #[case::organization("user_api_key_org_alias")]
    #[case::hash("user_api_key_hash")]
    fn client_cannot_forge_reserved_identity(
        mut input: BedrockRequestMetadataInput,
        #[case] key: &str,
    ) {
        input.sources[0].spend_logs = vec![(key.into(), "forged".into())];
        assert_eq!(
            resolve_bedrock_request_metadata(&input),
            input.sources[0].identity
        );
    }

    #[rstest]
    fn invalid_identity_is_dropped_without_rewriting(mut input: BedrockRequestMetadataInput) {
        input.sources[0].identity = vec![
            ("user_api_key_alias".into(), "prod-key".into()),
            ("user_api_key_team_alias".into(), "O'Brien's team".into()),
            ("user_api_key_user_email".into(), "x".repeat(300)),
        ];
        input.sources[0].spend_logs.clear();
        assert_eq!(
            resolve_bedrock_request_metadata(&input),
            [("user_api_key_alias".into(), "prod-key".into())]
        );
    }

    #[rstest]
    #[case::not_allowed(false)]
    #[case::explicitly_allowed(true)]
    fn email_requires_its_own_allowlist_entry(
        mut input: BedrockRequestMetadataInput,
        #[case] allowed: bool,
    ) {
        input
            .allowed_fields
            .retain(|field| allowed || field != "user_api_key_user_email");
        input.sources[0]
            .identity
            .push(("user_api_key_user_email".into(), "owner@example.com".into()));
        let resolved = resolve_bedrock_request_metadata(&input);
        assert_eq!(
            resolved
                .iter()
                .find(|(key, _)| key == "user_api_key_user_email")
                .map(|(_, value)| value.as_str()),
            allowed.then_some("owner@example.com")
        );
    }

    #[rstest]
    #[case::canonical(BEDROCK_REQUEST_METADATA_HEADER)]
    #[case::lowercase("x-amzn-bedrock-request-metadata")]
    #[case::mixed("x-AMZN-bedrock-Request-METADATA")]
    fn caller_header_is_replaced_in_every_casing(
        input: BedrockRequestMetadataInput,
        #[case] name: &str,
    ) {
        let headers = bedrock_request_metadata_headers(
            vec![
                (name.into(), "FORGED".into()),
                ("x-other".into(), "keep".into()),
            ],
            &input,
        )
        .unwrap();
        assert_eq!(headers[0], ("x-other".into(), "keep".into()));
        assert_eq!(headers.len(), 2);
        assert_eq!(
            serde_json::from_str::<Value>(&headers[1].1).unwrap(),
            json!({"user_api_key_alias":"prod-key", "user_api_key_team_alias":"platform", "cost_center":"cc-1"})
        );
    }

    #[rstest]
    #[case::empty_sources(vec![])]
    #[case::invalid_sources(vec![BedrockMetadataSource { identity: vec![("user_api_key_alias".into(), "O'Brien".into())], spend_logs: vec![] }])]
    fn owned_empty_metadata_evicts_the_caller(
        mut input: BedrockRequestMetadataInput,
        #[case] sources: Vec<BedrockMetadataSource>,
    ) {
        input.sources = sources;
        assert!(resolve_bedrock_request_metadata(&input).is_empty());
        assert!(
            bedrock_request_metadata_headers(
                vec![(BEDROCK_REQUEST_METADATA_HEADER.into(), "FORGED".into())],
                &input
            )
            .unwrap()
            .is_empty()
        );
    }

    #[rstest]
    fn disabled_forwarding_keeps_the_caller_header(input: BedrockRequestMetadataInput) {
        let off = BedrockRequestMetadataInput {
            allowed_fields: vec![],
            ..input
        };
        let headers = vec![(BEDROCK_REQUEST_METADATA_HEADER.into(), "caller-set".into())];
        assert_eq!(
            bedrock_request_metadata_headers(headers.clone(), &off).unwrap(),
            headers
        );
        assert!(resolve_bedrock_request_metadata(&off).is_empty());
    }

    fn fixed_clock() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    #[rstest]
    fn resolved_metadata_header_is_signed(input: BedrockRequestMetadataInput) {
        let headers = bedrock_request_metadata_headers(vec![], &input).unwrap();
        let signer = SigV4Signer::new(
            "us-east-1".into(),
            BEDROCK_SERVICE,
            litellm_auth_aws::Credentials::new("test", "secret", None, None, "test"),
        )
        .with_clock(fixed_clock);
        let sent = OutboundRequest::signed_json(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke".into(),
            headers,
            &json!({"messages":[]}),
            None,
            &signer,
        )
        .unwrap();
        assert!(
            sent.header("Authorization")
                .unwrap()
                .contains("x-amzn-bedrock-request-metadata")
        );
        assert_eq!(
            serde_json::from_str::<Value>(sent.header(BEDROCK_REQUEST_METADATA_HEADER).unwrap())
                .unwrap(),
            json!({"user_api_key_alias":"prod-key", "user_api_key_team_alias":"platform", "cost_center":"cc-1"})
        );
    }
}
