use litellm_auth::{
    CredentialPlacement, CredentialPlanKind, CredentialRule, ExistingHeaderBehavior,
    ProviderAuthPolicy, ResolvedCredential, SecretValue,
};

const RULES: &[CredentialRule] = &[CredentialRule {
    kind: CredentialPlanKind::Static,
    placement: CredentialPlacement::Header("x-api-key"),
}];

#[test]
fn facade_applies_shared_auth_policy() {
    let policy = ProviderAuthPolicy {
        rules: RULES,
        accepted_existing_headers: &["x-api-key"],
        existing_header_behavior: ExistingHeaderBehavior::Preserve,
        scope: None,
        audience: None,
    };

    let headers = policy
        .apply(
            Vec::new(),
            CredentialPlanKind::Static,
            &ResolvedCredential::Static(SecretValue::new("secret")),
        )
        .expect("facade policy applies");

    assert_eq!(
        headers,
        vec![("x-api-key".to_string(), "secret".to_string())]
    );
}
