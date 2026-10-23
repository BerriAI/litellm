use http::{HeaderMap, HeaderName, HeaderValue};

use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CredentialPlacement {
    Bearer,
    Header(&'static str),
}

impl CredentialPlacement {
    pub const fn header_name(self) -> &'static str {
        match self {
            Self::Bearer => "authorization",
            Self::Header(name) => name,
        }
    }

    pub fn header(self, credential: &str) -> Result<(HeaderName, HeaderValue), Error> {
        let name = HeaderName::try_from(self.header_name()).map_err(|_| Error::InvalidHeader)?;
        let value = match self {
            Self::Bearer => HeaderValue::try_from(format!("Bearer {credential}")),
            Self::Header(_) => HeaderValue::try_from(credential),
        };
        let mut value = value.map_err(|_| Error::InvalidHeader)?;
        value.set_sensitive(true);
        Ok((name, value))
    }
}

pub fn apply_credential(
    mut headers: HeaderMap,
    credential: &str,
    placement: CredentialPlacement,
) -> Result<HeaderMap, Error> {
    if credential.trim().is_empty() {
        return Err(Error::InvalidConfiguration(crate::ErrorDetail::Empty {
            subject: "credential",
        }));
    }
    let (name, value) = placement.header(credential)?;
    if headers.contains_key(&name) {
        return Err(Error::InvalidConfiguration(
            crate::ErrorDetail::DuplicateHeader(placement.header_name()),
        ));
    }
    headers.insert(name, value);
    Ok(headers)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn bearer_uses_authorization_header() {
        let headers = apply_credential(HeaderMap::new(), "key", CredentialPlacement::Bearer)
            .expect("credential applies");

        assert_eq!(headers.len(), 1);
        assert_eq!(headers["Authorization"], "Bearer key");
        assert!(headers["Authorization"].is_sensitive());
    }

    #[test]
    fn named_header_rejects_existing_value_in_any_case() {
        let error = apply_credential(
            HeaderMap::from_iter([(
                HeaderName::from_static("ocp-apim-subscription-key"),
                HeaderValue::from_static("caller-key"),
            )]),
            "configured-key",
            CredentialPlacement::Header("Ocp-Apim-Subscription-Key"),
        )
        .expect_err("provider policy must handle existing credentials");

        assert!(
            litellm_error::Report::new(&error)
                .to_string()
                .contains("already exists")
        );
    }

    #[rstest]
    #[case::newline_in_credential(CredentialPlacement::Bearer, "key\r\nx-injected: 1")]
    #[case::invalid_header_name(CredentialPlacement::Header("x api key"), "key")]
    fn a_credential_that_cannot_be_a_header_is_rejected(
        #[case] placement: CredentialPlacement,
        #[case] credential: &str,
    ) {
        assert_eq!(
            apply_credential(HeaderMap::new(), credential, placement),
            Err(Error::InvalidHeader)
        );
    }
}
