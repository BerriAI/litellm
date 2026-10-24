use http::HeaderMap;

use crate::http::apply_credential;
use crate::{CredentialPlacement, Error, ErrorDetail, ResolvedCredential};

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
        self.existing_credential_header(headers).is_some()
    }

    fn existing_credential_header(&self, headers: &HeaderMap) -> Option<&'static str> {
        self.accepted_existing_headers
            .iter()
            .copied()
            .find(|accepted| headers.contains_key(*accepted))
    }

    pub fn apply(
        &self,
        headers: HeaderMap,
        kind: CredentialPlanKind,
        credential: &ResolvedCredential,
    ) -> Result<HeaderMap, Error> {
        if let Some(existing) = self.existing_credential_header(&headers) {
            return match self.existing_header_behavior {
                ExistingHeaderBehavior::Preserve => Ok(headers),
                ExistingHeaderBehavior::Reject => Err(Error::InvalidConfiguration(
                    ErrorDetail::DuplicateHeader(existing),
                )),
            };
        }
        let rule = self
            .rules
            .iter()
            .find(|rule| rule.kind == kind)
            .ok_or(Error::InvalidConfiguration(ErrorDetail::PlanNotAllowed))?;
        apply_credential(headers, credential.secret().expose(), rule.placement)
    }
}

#[cfg(test)]
mod tests {
    use super::{CredentialPlanKind, CredentialRule, ExistingHeaderBehavior, ProviderAuthPolicy};
    use crate::{CredentialPlacement, ResolvedCredential, SecretValue};

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
    fn rules_define_allowed_plans_and_credential_placement() {
        let headers = POLICY
            .apply(
                http::HeaderMap::new(),
                CredentialPlanKind::Static,
                &ResolvedCredential::Static(SecretValue::new("secret")),
            )
            .unwrap();

        assert_eq!(headers.len(), 1);
        assert_eq!(headers["x-api-key"], "secret");
    }

    #[test]
    fn unsupported_plan_is_rejected() {
        let error = POLICY
            .apply(
                http::HeaderMap::new(),
                CredentialPlanKind::Entra,
                &ResolvedCredential::Static(SecretValue::new("secret")),
            )
            .unwrap_err();

        assert!(
            litellm_error::Report::new(&error)
                .to_string()
                .contains("not allowed")
        );
    }
}
