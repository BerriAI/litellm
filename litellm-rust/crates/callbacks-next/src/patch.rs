use litellm_host::event::WireRequest;
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::redact::is_credential_header;

#[derive(Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WirePatch {
    #[serde(default)]
    pub headers: HeaderPatch,
    #[serde(default)]
    pub body: Option<Value>,
}

#[derive(Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeaderPatch {
    #[serde(default)]
    pub remove: Vec<String>,
    #[serde(default)]
    pub set: Vec<(String, String)>,
}

#[derive(Debug, PartialEq, Error)]
pub enum PatchError {
    #[error("protected credential header: {0}")]
    ProtectedHeader(String),
    #[error("invalid header name: {0}")]
    InvalidHeaderName(String),
    #[error("invalid header value for: {0}")]
    InvalidHeaderValue(String),
}

fn valid_header_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

fn validate_name(name: &str) -> Result<(), PatchError> {
    if is_credential_header(name) {
        return Err(PatchError::ProtectedHeader(name.to_string()));
    }
    if !valid_header_name(name) {
        return Err(PatchError::InvalidHeaderName(name.to_string()));
    }
    Ok(())
}

pub fn apply(wire: WireRequest, patch: WirePatch) -> Result<WireRequest, PatchError> {
    for name in &patch.headers.remove {
        validate_name(name)?;
    }
    for (name, value) in &patch.headers.set {
        validate_name(name)?;
        if value
            .bytes()
            .any(|byte| matches!(byte, b'\r' | b'\n' | b'\0'))
        {
            return Err(PatchError::InvalidHeaderValue(name.clone()));
        }
    }

    let HeaderPatch { remove, set } = patch.headers;
    let retained: Vec<(String, String)> = wire
        .headers
        .into_iter()
        .filter(|(name, _)| {
            !remove
                .iter()
                .chain(set.iter().map(|(name, _)| name))
                .any(|changed| name.eq_ignore_ascii_case(changed))
        })
        .collect();
    let headers = retained.into_iter().chain(set).collect();
    Ok(WireRequest {
        url: wire.url,
        headers,
        body: patch.body.unwrap_or(wire.body),
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn wire() -> WireRequest {
        WireRequest {
            url: "https://provider.example/v1".to_string(),
            headers: vec![
                ("Authorization".to_string(), "secret".to_string()),
                ("X-Trace".to_string(), "one".to_string()),
                ("x-trace".to_string(), "two".to_string()),
                ("Keep".to_string(), "yes".to_string()),
            ],
            body: json!({"old": true}),
        }
    }

    #[test]
    fn default_patch_is_identity() {
        let original = wire();
        assert_eq!(apply(original.clone(), WirePatch::default()), Ok(original));
    }

    #[test]
    fn remove_precedes_set_and_names_match_case_insensitively() {
        let patch = WirePatch {
            headers: HeaderPatch {
                remove: vec!["keep".to_string()],
                set: vec![("X-TRACE".to_string(), "three".to_string())],
            },
            body: Some(json!({"new": true})),
        };
        let result = apply(wire(), patch).unwrap();
        assert_eq!(
            result.headers,
            vec![
                ("Authorization".to_string(), "secret".to_string()),
                ("X-TRACE".to_string(), "three".to_string()),
            ]
        );
        assert_eq!(result.body, json!({"new": true}));
        assert_eq!(result.url, "https://provider.example/v1");
    }

    #[rstest]
    #[case(json!({"url": "https://attacker.example"}))]
    #[case(json!({"headers": {"append": []}}))]
    fn unknown_patch_fields_are_rejected(#[case] value: Value) {
        assert!(serde_json::from_value::<WirePatch>(value).is_err());
    }

    #[rstest]
    #[case("Authorization", "value", PatchError::ProtectedHeader("Authorization".to_string()))]
    #[case("bad name", "value", PatchError::InvalidHeaderName("bad name".to_string()))]
    #[case("X-Test", "bad\r\nvalue", PatchError::InvalidHeaderValue("X-Test".to_string()))]
    fn invalid_set_is_rejected(
        #[case] name: &str,
        #[case] value: &str,
        #[case] expected: PatchError,
    ) {
        let patch = WirePatch {
            headers: HeaderPatch {
                remove: vec![],
                set: vec![(name.to_string(), value.to_string())],
            },
            body: None,
        };
        assert_eq!(apply(wire(), patch), Err(expected));
    }

    #[test]
    fn credential_headers_cannot_be_removed() {
        let patch = WirePatch {
            headers: HeaderPatch {
                remove: vec!["AUTHORIZATION".to_string()],
                set: vec![],
            },
            body: None,
        };
        assert_eq!(
            apply(wire(), patch),
            Err(PatchError::ProtectedHeader("AUTHORIZATION".to_string()))
        );
    }

    proptest! {
        #[test]
        fn successful_patches_preserve_url(value in "[A-Za-z0-9]{0,20}") {
            let original = wire();
            let patch = WirePatch {
                headers: HeaderPatch { remove: vec![], set: vec![("X-Safe".to_string(), value)] },
                body: None,
            };
            let result = apply(original.clone(), patch).unwrap();
            prop_assert_eq!(result.url, original.url);
            prop_assert_eq!(result.headers[0].clone(), original.headers[0].clone());
        }
    }
}
