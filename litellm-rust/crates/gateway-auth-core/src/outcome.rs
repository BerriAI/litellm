use std::time::SystemTime;

use litellm_secret_value::SecretValue;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub enum Grant {
    NoMasterKey { token: Option<SecretValue> },
    UserAuthRoute,
    MasterKey,
    CachedAdminKey,
    VirtualKey { budget_throttle_pct: Option<f64> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelBudgetExceeded {
    pub message: String,
    pub current_cost: f64,
    pub max_budget: f64,
    pub entity_type: Option<String>,
    pub entity_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Denial {
    MissingKey,
    MalformedKey,
    UserAuthDisabled,
    ExpiredKey {
        expires: SystemTime,
        now: SystemTime,
        key_label: SecretValue,
    },
    MasterKeyOnlyRoute,
    NoDatabase,
    NotAVirtualKey,
    KeyBlocked,
    OwnerDeactivated {
        user_id: String,
    },
    TeamMemberBudgetExceeded {
        user_id: String,
        team_id: String,
        spend: f64,
        max_budget: f64,
    },
    KeyBudgetExceeded {
        spend: f64,
        max_budget: f64,
    },
    KeyWithoutToken,
    ModelBudgetExceeded(ModelBudgetExceeded),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Allow(Grant),
    Deny(Denial),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WithinModelBudgets;

pub type ModelBudgetResult = Result<WithinModelBudgets, ModelBudgetExceeded>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::VariantArray)]
pub enum AnonymousReason {
    PublicRoute,
    PassThroughAuthOff,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Anonymous {
    pub reason: AnonymousReason,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JwtIdentity {
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub end_user_id: Option<String>,
    pub role: Option<String>,
    pub scopes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct OAuth2Identity {
    pub token: String,
    pub user_id: Value,
    pub role: Value,
    pub team_id: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct OAuth2ProxyIdentity {
    pub user_id: Option<String>,
    pub user_email: Option<String>,
    pub team_id: Option<String>,
    pub team_alias: Option<String>,
    pub org_id: Option<String>,
    pub models: Option<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomIdentity {
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub hashed_token: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CustomAuthResult {
    Identity(CustomIdentity),
    ReplacedKey { token: String },
    NoIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Identity {
    MasterKey,
    Key { hashed_token: String },
    Jwt(JwtIdentity),
    OAuth2(OAuth2Identity),
    OAuth2Proxy(OAuth2ProxyIdentity),
    Custom(CustomIdentity),
    Anonymous(Anonymous),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::VariantArray)]
pub enum OAuth2Denial {
    InactiveToken,
    IntrospectionNotConfigured,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OAuth2ProxyDenial {
    TrustedProxyRangesMissing,
    UntrustedProxy { client_ip: Option<String> },
    ProxyMappingsMissing,
    DisallowedProxyMappings { fields: Vec<String> },
}

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
