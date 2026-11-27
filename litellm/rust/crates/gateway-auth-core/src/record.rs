//! The rows a caller belongs to, as the core reads them: one record per entity, holding its id and
//! what its stored row says, and nothing a request or another row adds. A host builds a record
//! from the row it read and decides nothing while it does. A record grows a field with the first
//! rule that reads it.

use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UserRecord {
    pub user_id: String,
    pub user_alias: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeamRecord {
    pub team_id: String,
    pub team_alias: Option<String>,
}

/// A user's row in a team.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MembershipRecord {
    pub user_id: String,
    pub team_id: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct OrgRecord {
    pub organization_id: String,
    pub organization_alias: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProjectRecord {
    pub project_id: String,
    pub project_alias: Option<String>,
    pub metadata: Option<Map<String, Value>>,
}

/// The request's end user. `found` is whether it has a row, so a request that names none, or one
/// with no row, reads an empty record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EndUserRecord {
    pub end_user_id: Option<String>,
    pub found: bool,
}
