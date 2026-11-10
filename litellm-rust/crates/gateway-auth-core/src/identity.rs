use std::time::SystemTime;

use litellm_secret_value::SecretValue;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::answers::KeyFacts;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::VariantArray)]
pub enum AnonymousReason {
    PublicRoute,
    PassThroughAuthOff,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Anonymous {
    pub reason: AnonymousReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JwtIdentity {
    ProxyAdmin,
    User,
    Key,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VirtualKey {
    pub key: KeyFacts,
}

#[derive(Clone, Debug, PartialEq)]
pub enum KeyIdentity {
    NoMasterKey { token: Option<SecretValue> },
    UserAuthRoute,
    MasterKey,
    CachedAdminKey,
    VirtualKey(VirtualKey),
}

#[derive(Clone, Debug, PartialEq)]
pub enum KeyDenial {
    MissingKey,
    MalformedKey,
    NoDatabase,
    NotAVirtualKey,
    KeyBlocked,
    OwnerDeactivated {
        user_id: String,
    },
    ExpiredKey {
        expires: SystemTime,
        now: SystemTime,
        key_label: SecretValue,
    },
    KeyWithoutToken,
}

pub type KeyAuthentication = Result<KeyIdentity, KeyDenial>;

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

#[derive(Clone, Debug, PartialEq)]
pub enum Identity {
    Key(KeyIdentity),
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

#[derive(Clone, Debug, PartialEq)]
pub enum AuthDenial {
    Key(KeyDenial),
    Jwt(JwtDenial),
    OAuth2(OAuth2Denial),
    OAuth2Proxy(OAuth2ProxyDenial),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JwtDenial {
    ManagedAgentNeedsDirectJwt,
    OwnerDeactivated { user_id: Option<String> },
    KeyMappingFailed { reason: String },
}

pub type JwtAuthentication = Result<JwtIdentity, JwtDenial>;
