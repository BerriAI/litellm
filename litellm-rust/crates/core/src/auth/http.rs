use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

use crate::AuthError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CredentialPlacement {
    Bearer,
    Header(&'static str),
}

impl CredentialPlacement {
    pub fn header_name(self) -> &'static str {
        match self {
            Self::Bearer => "Authorization",
            Self::Header(name) => name,
        }
    }
}

pub(crate) fn apply_credential(
    mut headers: HeaderMap,
    credential: &str,
    placement: CredentialPlacement,
) -> Result<HeaderMap, AuthError> {
    if credential.trim().is_empty() {
        return Err(AuthError::EmptyCredential);
    }
    let name = HeaderName::from_bytes(placement.header_name().as_bytes()).map_err(|_| {
        AuthError::InvalidCredentialHeaderName {
            name: placement.header_name(),
        }
    })?;
    if headers.contains_key(&name) {
        return Err(AuthError::CredentialHeaderAlreadyExists {
            name: placement.header_name(),
        });
    }
    let mut value = match placement {
        CredentialPlacement::Bearer => HeaderValue::from_str(&format!("Bearer {credential}")),
        CredentialPlacement::Header(_) => HeaderValue::from_str(credential),
    }
    .map_err(|_| AuthError::InvalidCredentialHeaderValue)?;
    value.set_sensitive(true);
    headers.insert(name, value);
    Ok(headers)
}

#[cfg(test)]
mod tests {
    use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};

    use super::{CredentialPlacement, apply_credential};
    use crate::AuthError;

    #[test]
    fn credential_headers_are_sensitive_in_built_requests() {
        for placement in [
            CredentialPlacement::Bearer,
            CredentialPlacement::Header("x-api-key"),
        ] {
            let headers = apply_credential(HeaderMap::new(), "secret", placement).unwrap();
            let request = reqwest::Client::new()
                .get("https://example.com")
                .headers(headers)
                .build()
                .unwrap();
            let value = &request.headers()[placement.header_name()];
            assert!(value.is_sensitive());
            assert!(!format!("{request:?}").contains("secret"));
            assert_eq!(
                value,
                match placement {
                    CredentialPlacement::Bearer => "Bearer secret",
                    _ => "secret",
                }
            );
        }
    }

    #[test]
    fn invalid_credentials_fail_with_typed_errors() {
        for placement in [
            CredentialPlacement::Bearer,
            CredentialPlacement::Header("x-api-key"),
        ] {
            for credential in ["", "   "] {
                assert_eq!(
                    apply_credential(HeaderMap::new(), credential, placement),
                    Err(AuthError::EmptyCredential)
                );
            }
            for credential in ["secret\r\nx-injected: true", "secret\0"] {
                assert_eq!(
                    apply_credential(HeaderMap::new(), credential, placement),
                    Err(AuthError::InvalidCredentialHeaderValue)
                );
            }
        }
    }

    #[test]
    fn duplicate_auth_is_rejected_case_insensitively() {
        let headers =
            HeaderMap::from_iter([(AUTHORIZATION, HeaderValue::from_static("Bearer caller"))]);
        assert_eq!(
            apply_credential(headers, "configured", CredentialPlacement::Bearer),
            Err(AuthError::CredentialHeaderAlreadyExists {
                name: "Authorization"
            })
        );
    }

    #[test]
    fn invalid_header_names_fail_with_a_typed_error() {
        assert_eq!(
            apply_credential(
                HeaderMap::new(),
                "secret",
                CredentialPlacement::Header("invalid name")
            ),
            Err(AuthError::InvalidCredentialHeaderName {
                name: "invalid name"
            })
        );
    }
}
