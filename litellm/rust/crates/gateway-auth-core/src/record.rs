//! The rows a caller belongs to, as the core reads them: one record per entity, holding what its
//! stored row says and nothing a request or another row adds. A host builds a record from the row
//! it read and decides nothing while it does. A record grows a field with the first rule that
//! reads it.

use serde_json::{Map, Value};

/// A team's row, as far as [`projection`](crate::projection) reads it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeamRecord {
    pub team_alias: Option<String>,
    pub models: Vec<String>,
    pub blocked: bool,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
    pub tpd_limit: Option<i64>,
    pub max_budget: Option<f64>,
    pub soft_budget: Option<f64>,
    pub model_max_budget: Option<Map<String, Value>>,
    pub spend: Option<f64>,
    pub metadata: Option<Map<String, Value>>,
    /// `model_aliases` of the team's alias table as its column holds it: an object, or the JSON
    /// text of one, which is how the proxy's writers store it. `None` for a team with no table.
    pub model_aliases: Option<Value>,
    pub object_permission_id: Option<String>,
    pub members_with_roles: Vec<Member>,
}

/// One entry of a team's `members_with_roles`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub user_id: Option<String>,
    pub user_email: Option<String>,
    pub role: MemberRole,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberRole {
    Admin,
    User,
}
