//! The rows a caller belongs to, as the core reads them: one record per entity, holding its id and
//! what its stored row says, and nothing a request or another row adds. A host builds a record
//! from the row it read and decides nothing while it does. A record grows a field with the first
//! rule that reads it, except a team's and a user's, which copy each column of their row that a
//! plain value holds, read or not. The host refuses a caller whose team or user holds one nothing
//! reads yet.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::authz::ModelList;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UserRecord {
    pub user_id: String,
    pub user_alias: Option<String>,
    pub user_email: Option<String>,
    pub user_role: Option<String>,
    pub teams: Vec<String>,
    pub team_id: Option<String>,
    pub organization_id: Option<String>,
    pub models: Vec<String>,
    pub limits: MinuteLimits,
    pub max_parallel_requests: Option<i64>,
    pub budget: Budget,
    pub rollover_max_budget: Option<f64>,
    pub rollover_credit: Option<f64>,
    pub budget_duration: Option<String>,
    pub allowed_cache_controls: Vec<String>,
    pub policies: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeamRecord {
    pub team_id: String,
    pub team_alias: Option<String>,
    /// The models the team may call, which the model rule reads as the team's layer.
    pub models: Vec<String>,
    pub organization_id: Option<String>,
    pub limits: MinuteLimits,
    pub tpd_limit: Option<i64>,
    pub max_parallel_requests: Option<i64>,
    pub budget: Budget,
    pub soft_budget: Option<f64>,
    pub budget_duration: Option<String>,
    pub blocked: bool,
    pub access_group_ids: Vec<String>,
    pub default_team_member_models: Vec<String>,
    pub policies: Vec<String>,
    pub allow_team_guardrail_config: bool,
    pub model_rpm_limit: BTreeMap<String, i64>,
    pub model_tpm_limit: BTreeMap<String, i64>,
    pub priority: Option<String>,
    /// The `max_budget` of the budget `metadata.team_member_budget_id` names, which a member with
    /// no budget of its own is held to.
    pub member_max_budget: Option<f64>,
}

/// A user's row in a team. Its limits and budget are its budget row's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MembershipRecord {
    pub user_id: String,
    pub team_id: String,
    pub limits: MinuteLimits,
    pub budget: Budget,
}

/// An organization. Its limits and budget are its budget row's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OrgRecord {
    pub organization_id: String,
    pub organization_alias: Option<String>,
    pub limits: MinuteLimits,
    pub budget: Budget,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProjectRecord {
    pub project_id: String,
    pub project_alias: Option<String>,
    pub metadata: Option<Map<String, Value>>,
    pub budget: Budget,
}

/// The request's end user. `found` is whether it has a row, so a request that names none, or one
/// with no row, reads an empty record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EndUserRecord {
    pub end_user_id: Option<String>,
    pub found: bool,
    pub limits: MinuteLimits,
    pub budget: Budget,
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

/// The `max_budget` a row holds and what it has spent against it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Budget {
    pub max: Option<f64>,
    pub spend: f64,
}

/// What a caller is known by: the id it names, and the alias only a stored row gives.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Label {
    pub id: String,
    pub alias: Option<String>,
}

/// A record that stands for something a caller names by id.
pub trait Record {
    fn id(&self) -> &str;
    fn alias(&self) -> Option<&str>;

    fn label(&self) -> Label {
        Label {
            id: self.id().to_owned(),
            alias: self.alias().map(str::to_owned),
        }
    }
}

impl Record for UserRecord {
    fn id(&self) -> &str {
        &self.user_id
    }

    fn alias(&self) -> Option<&str> {
        self.user_alias.as_deref()
    }
}

impl TeamRecord {
    pub fn model_list(&self) -> ModelList<'_> {
        ModelList::read(&self.models)
    }
}

impl Record for TeamRecord {
    fn id(&self) -> &str {
        &self.team_id
    }

    fn alias(&self) -> Option<&str> {
        self.team_alias.as_deref()
    }
}

impl Record for OrgRecord {
    fn id(&self) -> &str {
        &self.organization_id
    }

    fn alias(&self) -> Option<&str> {
        self.organization_alias.as_deref()
    }
}

impl Record for ProjectRecord {
    fn id(&self) -> &str {
        &self.project_id
    }

    fn alias(&self) -> Option<&str> {
        self.project_alias.as_deref()
    }
}
