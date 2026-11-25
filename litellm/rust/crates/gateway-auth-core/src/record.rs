//! The rows a caller belongs to, as the core reads them: one record per entity, holding its id and
//! what its stored row says, and nothing a request or another row adds. A host builds a record
//! from the row it read and decides nothing while it does. A record grows a field with the first
//! rule that reads it.

use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UserRecord {
    pub user_id: String,
    pub user_alias: Option<String>,
    pub limits: MinuteLimits,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeamRecord {
    pub team_id: String,
    pub team_alias: Option<String>,
    pub limits: MinuteLimits,
}

/// A user's row in a team. Its limits are its budget row's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MembershipRecord {
    pub user_id: String,
    pub team_id: String,
    pub limits: MinuteLimits,
}

/// An organization. Its limits are its budget row's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OrgRecord {
    pub organization_id: String,
    pub organization_alias: Option<String>,
    pub limits: MinuteLimits,
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
    /// Its budget row's.
    pub limits: MinuteLimits,
}

/// The requests and tokens per minute a row allows, `None` where it sets no limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MinuteLimits {
    pub rpm: Option<i64>,
    pub tpm: Option<i64>,
}

impl MinuteLimits {
    pub fn counts_tokens(self) -> bool {
        self.tpm.is_some()
    }
}

/// A record that stands for something a caller names by id.
pub trait Record {
    fn id(&self) -> &str;
}

impl Record for UserRecord {
    fn id(&self) -> &str {
        &self.user_id
    }
}

impl Record for TeamRecord {
    fn id(&self) -> &str {
        &self.team_id
    }
}

impl Record for OrgRecord {
    fn id(&self) -> &str {
        &self.organization_id
    }
}

impl Record for ProjectRecord {
    fn id(&self) -> &str {
        &self.project_id
    }
}
