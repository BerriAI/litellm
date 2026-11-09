use litellm_secret_value::SecretValue;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Default)]
pub struct PresentedKeys {
    pub custom_header: Option<SecretValue>,
    pub litellm_header: Option<SecretValue>,
    pub authorization: Option<SecretValue>,
    pub api_key: Option<SecretValue>,
    pub x_api_key: Option<SecretValue>,
    pub google: Option<SecretValue>,
    pub apim: Option<SecretValue>,
    pub query: Option<SecretValue>,
    pub pass_through: Option<SecretValue>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, strum::VariantArray)]
pub enum KeySource {
    #[serde(rename = "custom_header")]
    CustomHeader,
    #[serde(rename = "x-litellm-api-key")]
    LitellmHeader,
    #[serde(rename = "authorization")]
    Authorization,
    #[serde(rename = "api-key")]
    ApiKey,
    #[serde(rename = "x-api-key")]
    XApiKey,
    #[serde(rename = "x-goog-api-key")]
    Google,
    #[serde(rename = "ocp-apim-subscription-key")]
    Apim,
    #[serde(rename = "key")]
    Query,
    #[serde(rename = "pass_through_endpoint")]
    PassThrough,
}

#[derive(veil::Redact, Serialize)]
pub struct SelectedKey {
    pub source: KeySource,
    #[redact(with = "[REDACTED]")]
    pub presented: String,
    #[redact(with = "[REDACTED]")]
    pub token: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct WebSocketKeys {
    pub authorization: Option<SecretValue>,
    pub api_key: Option<SecretValue>,
    pub subprotocols: Option<SecretValue>,
}

#[derive(Debug, Serialize, PartialEq, Eq, strum::VariantArray)]
pub enum WebSocketDenial {
    #[serde(rename = "no_api_key")]
    NoApiKey,
    #[serde(rename = "invalid_authorization")]
    InvalidAuthorization,
}

#[derive(veil::Redact, Serialize)]
#[serde(tag = "kind", content = "value")]
pub enum WebSocketResult {
    Token(#[redact(with = "[REDACTED]")] String),
    Denial(WebSocketDenial),
}

fn whitespace(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '\u{1c}'..='\u{1f}')
}

fn credential_part(value: &str) -> Option<&str> {
    let end = value
        .find(|ch| ch == '/' || ch == ',' || whitespace(ch))
        .unwrap_or(value.len());
    (end != 0).then_some(&value[..end])
}

pub fn scheme_credential(value: &str) -> Option<String> {
    if value.starts_with("AWS4-HMAC-SHA256") {
        let bearer = value
            .match_indices("Credential=Bearer")
            .find_map(|(index, prefix)| {
                let tail = &value[index + prefix.len()..];
                if !tail.starts_with(whitespace) {
                    return None;
                }
                credential_part(tail.trim_start_matches(whitespace))
            });
        return bearer
            .or_else(|| {
                value
                    .match_indices("Credential=")
                    .find_map(|(index, prefix)| credential_part(&value[index + prefix.len()..]))
            })
            .map(str::to_owned);
    }
    let (scheme, token) = value.split_once(' ')?;
    matches!(scheme.to_lowercase().as_str(), "bearer" | "basic").then(|| token.to_owned())
}

pub fn litellm_header_token(value: &str) -> String {
    scheme_credential(value).unwrap_or_else(|| value.to_owned())
}

pub fn select_api_key(keys: PresentedKeys) -> Option<SelectedKey> {
    if let Some(value) = keys.custom_header {
        return Some(SelectedKey {
            source: KeySource::CustomHeader,
            presented: value.expose().to_owned(),
            token: scheme_credential(value.expose()).unwrap_or_default(),
        });
    }
    if let Some(value) = keys.litellm_header {
        return Some(SelectedKey {
            source: KeySource::LitellmHeader,
            presented: value.expose().to_owned(),
            token: litellm_header_token(value.expose()),
        });
    }
    if let Some(value) = keys
        .authorization
        .filter(|value| !value.expose().is_empty())
    {
        return Some(SelectedKey {
            source: KeySource::Authorization,
            presented: value.expose().to_owned(),
            token: scheme_credential(value.expose()).unwrap_or_default(),
        });
    }
    [
        (KeySource::ApiKey, keys.api_key),
        (KeySource::XApiKey, keys.x_api_key),
        (KeySource::Google, keys.google),
        (KeySource::Apim, keys.apim),
        (
            KeySource::Query,
            keys.query.filter(|value| !value.expose().is_empty()),
        ),
        (KeySource::PassThrough, keys.pass_through),
    ]
    .into_iter()
    .find_map(|(source, value)| {
        value.map(|value| SelectedKey {
            source,
            presented: value.expose().to_owned(),
            token: value.expose().to_owned(),
        })
    })
}

pub fn select_websocket_api_key(keys: WebSocketKeys) -> WebSocketResult {
    if let Some(value) = keys
        .authorization
        .filter(|value| !value.expose().is_empty())
    {
        let Some((scheme, token)) = value.expose().split_once(' ') else {
            return WebSocketResult::Denial(WebSocketDenial::InvalidAuthorization);
        };
        if scheme.to_lowercase() != "bearer" {
            return WebSocketResult::Denial(WebSocketDenial::InvalidAuthorization);
        }
        return WebSocketResult::Token(token.trim_matches(whitespace).to_owned());
    }
    if let Some(value) = keys.api_key.filter(|value| !value.expose().is_empty()) {
        return WebSocketResult::Token(value.expose().to_owned());
    }
    let token = keys.subprotocols.as_ref().and_then(|value| {
        value.expose().split(',').find_map(|protocol| {
            protocol
                .trim_matches(whitespace)
                .strip_prefix("openai-insecure-api-key.")
        })
    });
    match token {
        Some(token) if !token.is_empty() => WebSocketResult::Token(token.to_owned()),
        _ => WebSocketResult::Denial(WebSocketDenial::NoApiKey),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::bearer("Bearer  token", Some(" token"))]
    #[case::empty("Bearer ", Some(""))]
    #[case::basic("bAsIc token", Some("token"))]
    #[case::unsupported("Token token", None)]
    #[case::aws("AWS4-HMAC-SHA256 Credential=Bearer\u{1c}token/date", Some("token"))]
    #[case::aws_precedence(
        "AWS4-HMAC-SHA256 Credential=first/date, Credential=Bearer second/date",
        Some("second")
    )]
    fn schemes(#[case] value: &str, #[case] expected: Option<&str>) {
        assert_eq!(scheme_credential(value).as_deref(), expected);
    }

    #[rstest]
    fn empty_custom_header_still_wins() {
        let selected = select_api_key(PresentedKeys {
            custom_header: Some("".into()),
            api_key: Some("other".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(selected.source, KeySource::CustomHeader);
        assert_eq!(selected.token, "");
    }

    #[rstest]
    fn secrets_are_redacted() {
        let keys = PresentedKeys {
            api_key: Some("credential-value".into()),
            ..Default::default()
        };
        let keys_debug = format!("{keys:?}");
        let debug = format!("{keys_debug} {:?}", select_api_key(keys).unwrap());
        assert!(!debug.contains("credential-value"));
        assert!(debug.contains("REDACTED"));
    }
}
