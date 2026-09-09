use reqwest::header::HeaderMap;

use crate::AuthError;

use super::http::apply_credential;
use super::{CredentialPlacement, ResolvedCredential};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CredentialPlanKind {
    Static,
    Entra,
    Caller,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CredentialRule {
    pub kind: CredentialPlanKind,
    pub placement: CredentialPlacement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExistingHeaderBehavior {
    Preserve,
    Reject,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderAuthPolicy {
    pub rules: &'static [CredentialRule],
    pub accepted_existing_headers: &'static [&'static str],
    pub existing_header_behavior: ExistingHeaderBehavior,
    pub scope: Option<&'static str>,
    pub audience: Option<&'static str>,
}

impl ProviderAuthPolicy {
    pub fn has_existing_credential(&self, headers: &HeaderMap) -> bool {
        self.accepted_existing_headers
            .iter()
            .any(|name| headers.contains_key(*name))
    }

    pub fn apply(
        &self,
        headers: HeaderMap,
        kind: CredentialPlanKind,
        credential: &ResolvedCredential,
    ) -> Result<HeaderMap, AuthError> {
        if self.has_existing_credential(&headers) {
            return match self.existing_header_behavior {
                ExistingHeaderBehavior::Preserve => Ok(headers),
                ExistingHeaderBehavior::Reject => Err(AuthError::ExistingCredentialHeader),
            };
        }
        let rule = self
            .rules
            .iter()
            .find(|rule| rule.kind == kind)
            .ok_or(AuthError::CredentialPlanNotAllowed)?;
        apply_credential(headers, credential.secret().expose(), rule.placement)
    }
}

#[cfg(test)]
mod tests {
    use super::{CredentialPlanKind, CredentialRule, ExistingHeaderBehavior, ProviderAuthPolicy};
    use crate::auth::{CredentialPlacement, ResolvedCredential, SecretValue};
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

    const RULES: &[CredentialRule] = &[CredentialRule {
        kind: CredentialPlanKind::Static,
        placement: CredentialPlacement::Header("x-api-key"),
    }];
    const POLICY: ProviderAuthPolicy = ProviderAuthPolicy {
        rules: RULES,
        accepted_existing_headers: &["x-api-key"],
        existing_header_behavior: ExistingHeaderBehavior::Preserve,
        scope: None,
        audience: None,
    };

    #[test]
    fn existing_credential_policy_preserves_or_rejects_caller_auth() {
        let headers = HeaderMap::from_iter([(
            HeaderName::from_static("x-api-key"),
            HeaderValue::from_static("caller"),
        )]);
        let credential = ResolvedCredential::Static(SecretValue::new("configured"));
        assert_eq!(
            POLICY
                .apply(headers.clone(), CredentialPlanKind::Static, &credential)
                .unwrap()["x-api-key"],
            "caller"
        );
        let rejecting = ProviderAuthPolicy {
            existing_header_behavior: ExistingHeaderBehavior::Reject,
            ..POLICY
        };
        assert_eq!(
            rejecting.apply(headers, CredentialPlanKind::Static, &credential),
            Err(crate::AuthError::ExistingCredentialHeader)
        );
    }

    #[test]
    fn rules_define_allowed_plans_and_credential_placement() {
        let headers = POLICY
            .apply(
                HeaderMap::new(),
                CredentialPlanKind::Static,
                &ResolvedCredential::Static(SecretValue::new("secret")),
            )
            .unwrap();

        assert_eq!(
            headers,
            HeaderMap::from_iter([(
                HeaderName::from_static("x-api-key"),
                HeaderValue::from_static("secret")
            )])
        );
    }

    #[test]
    fn unsupported_plan_is_rejected() {
        let error = POLICY
            .apply(
                HeaderMap::new(),
                CredentialPlanKind::Entra,
                &ResolvedCredential::Static(SecretValue::new("secret")),
            )
            .unwrap_err();

        assert_eq!(error, crate::AuthError::CredentialPlanNotAllowed);
    }
}
