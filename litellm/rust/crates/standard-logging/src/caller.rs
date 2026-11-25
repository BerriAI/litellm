use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Budget {
    pub spend: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Named {
    pub id: Option<String>,
    pub alias: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Key {
    pub hash: Option<String>,
    pub alias: Option<String>,
    pub budget: Budget,
    pub budget_reset_at: Option<String>,
    pub metadata: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct User {
    pub id: Option<String>,
    pub email: Option<String>,
    pub budget: Budget,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Team {
    pub id: Option<String>,
    pub alias: Option<String>,
    pub budget: Budget,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Customer {
    pub member: Named,
    pub team: Named,
    pub organization: Named,
    pub budget_scopes: Option<String>,
    pub hierarchy: Option<Value>,
}

/// The app a customer hierarchy resolved the key to, which the proxy stamps as
/// `user_api_key_app_id`, `user_api_key_app_group_id` and `user_api_key_app_alias` only when the
/// hierarchy names both an app id and an alias
/// (`LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct App {
    pub id: String,
    /// Logged only when the app belongs to a group.
    pub group_id: Option<String>,
    pub alias: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Caller {
    pub key: Key,
    pub user: User,
    pub team: Team,
    pub organization: Named,
    pub project: Named,
    pub customer: Customer,
    pub app: Option<App>,
    pub end_user_id: Option<String>,
    pub request_route: Option<String>,
}
