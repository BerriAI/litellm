use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{Error, base_llm::auth::Headers};

pub const BEDROCK_REQUEST_METADATA_HEADER: &str = "X-Amzn-Bedrock-Request-Metadata";
pub const BEDROCK_REQUEST_METADATA_MAX_PAIRS: usize = 16;
const IDENTITY_PREFIX: &str = "user_api_key_";
const CLIENT_FIELD: &str = "spend_logs_metadata";

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BedrockMetadataSource {
    pub identity: Vec<(String, String)>,
    pub spend_logs: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
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

/// Serialize like Python's `json.dumps(..., separators=(",", ":"))`, which escapes every
/// non-ASCII char as `\uXXXX` (surrogate pairs above U+FFFF).
fn escape_non_ascii(json: &str) -> String {
    json.chars()
        .flat_map(|c| {
            if c.is_ascii() {
                vec![c]
            } else {
                let mut units = vec![0u16; 2];
                let encoded: &[u16] = c.encode_utf16(&mut units);
                let mut escaped: Vec<char> = Vec::with_capacity(6 * encoded.len());
                for unit in encoded {
                    escaped.extend(format!("\\u{unit:04x}").chars());
                }
                escaped
            }
        })
        .collect()
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
        Some(escape_non_ascii(&serde_json::to_string(&object).map_err(
            |error| {
                Error::InvalidRequest(crate::ErrorDetail::invalid(
                    "Bedrock request metadata",
                    error,
                ))
            },
        )?))
    };
    Ok(headers
        .into_iter()
        .filter(|(name, _)| !name.eq_ignore_ascii_case(BEDROCK_REQUEST_METADATA_HEADER))
        .chain(value.map(|value| (BEDROCK_REQUEST_METADATA_HEADER.into(), value)))
        .collect())
}
