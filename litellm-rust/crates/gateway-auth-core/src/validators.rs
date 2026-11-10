use crate::identity::AnonymousReason;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::VariantArray)]
pub enum OAuth2Mode {
    Introspection,
    Proxy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::VariantArray)]
pub enum PremiumFeature {
    OAuth2,
    Jwt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyPlan {
    Key { pass_through_key: bool },
    Anonymous(AnonymousReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CredentialPlan {
    Key(KeyPlan),
    Jwt,
    OAuth2(OAuth2Mode),
    PremiumRequired(PremiumFeature),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    Credential(CredentialPlan),
    Custom,
}
