use crate::AuthError;
use crate::auth::error::AuthConfigurationError;

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
    headers: Vec<(String, String)>,
    credential: &str,
    placement: CredentialPlacement,
) -> Result<Vec<(String, String)>, AuthError> {
    if credential.trim().is_empty() {
        return Err(AuthError::Configuration(
            AuthConfigurationError::EmptyCredential,
        ));
    }
    if headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case(placement.header_name()))
    {
        return Err(AuthError::Configuration(
            AuthConfigurationError::DuplicateHeader(placement.header_name()),
        ));
    }
    let value = match placement {
        CredentialPlacement::Bearer => format!("Bearer {credential}"),
        CredentialPlacement::Header(_) => credential.to_string(),
    };
    Ok(
        std::iter::once((placement.header_name().to_string(), value))
            .chain(headers)
            .collect(),
    )
}

/// How the upstream call is authenticated. API-key strategies are resolved in
/// `prepare`; SigV4 needs the serialized body, so the handler signs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestAuth {
    Header { name: &'static str, value: String },
    Bearer { token: String },
    AwsSigV4 { region: String },
}

#[cfg(test)]
mod tests {
    use super::{CredentialPlacement, apply_credential};

    #[test]
    fn bearer_uses_authorization_header() {
        let headers = apply_credential(Vec::new(), "key", CredentialPlacement::Bearer)
            .expect("credential applies");

        assert_eq!(
            headers,
            vec![("Authorization".to_string(), "Bearer key".to_string())]
        );
    }

    #[test]
    fn named_header_rejects_existing_value() {
        let error = apply_credential(
            vec![(
                "ocp-apim-subscription-key".to_string(),
                "caller-key".to_string(),
            )],
            "configured-key",
            CredentialPlacement::Header("Ocp-Apim-Subscription-Key"),
        )
        .expect_err("provider policy must handle existing credentials");

        assert!(error.to_string().contains("already exists"));
    }
}
