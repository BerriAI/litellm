use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// The core auth identity for an API key.
///
/// This is the Rust equivalent of Python's `UserAPIKeyAuth`, containing only
/// the fields needed for auth decisions. The full Python object has 40+ fields
/// from a 7-table JOIN; this covers the critical subset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyObject {
    /// The hashed token (64-char SHA-256 hex digest). Primary key.
    pub token: String,
    pub key_name: Option<String>,
    pub key_alias: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub project_id: Option<String>,
    pub agent_id: Option<String>,
    /// Cumulative spend in USD.
    pub spend: f64,
    /// Maximum budget in USD. None means unlimited.
    pub max_budget: Option<f64>,
    pub budget_duration: Option<String>,
    /// Allowed models. Empty set means all models allowed.
    pub models: HashSet<String>,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
    pub max_parallel_requests: Option<i64>,
    pub blocked: bool,
    /// Allowed routes. Empty set means all routes allowed.
    pub allowed_routes: HashSet<String>,
    pub metadata: Option<String>,
    /// Unix timestamp of last cache refresh.
    pub last_refreshed_at: Option<f64>,
    /// Whether this key has expired.
    pub expires: Option<String>,
}

impl KeyObject {
    pub fn is_expired(&self) -> bool {
        let Some(expires) = &self.expires else {
            return false;
        };
        parse_expiry(expires).is_some_and(|ts| ts < current_unix_timestamp())
    }

    pub fn has_model_access(&self, model: &str) -> bool {
        self.models.is_empty() || self.models.contains(model)
    }

    pub fn has_route_access(&self, route: &str) -> bool {
        self.allowed_routes.is_empty() || self.allowed_routes.contains(route)
    }

    pub fn is_within_budget(&self) -> bool {
        let Some(max_budget) = self.max_budget else {
            return true;
        };
        self.spend < max_budget
    }
}

fn current_unix_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn parse_expiry(expiry: &str) -> Option<f64> {
    expiry.parse::<f64>().ok()
}
