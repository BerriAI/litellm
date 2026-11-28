//! The rows a caller belongs to, as the core reads them: one record per entity, holding its id and
//! what its stored row says, and nothing a request or another row adds. A host builds a record
//! from the row it read and decides nothing while it does. A record grows a field with the first
//! rule that reads it, except for the plain values of a team, user, membership, organization,
//! project or end user, which are copied whether or not the gateway uses them. The host refuses a caller whose
//! association holds a column the gateway does not honor.

use std::collections::BTreeMap;

use jiff::Timestamp;
use litellm_router_types::RouterSettings;
use litellm_secret_value::SecretValue;
use serde::Deserialize;
use serde_with::{Map, serde_as};

use crate::authz::ModelList;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UserRecord {
    pub user_id: String,
    pub object_permission: Option<Box<ObjectPermissionRecord>>,
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
    pub budget_limits: Vec<BudgetWindow>,
    pub model_max_budget: BTreeMap<String, ModelBudget>,
    pub model_spend: BTreeMap<String, f64>,
    pub allowed_cache_controls: Vec<String>,
    pub policies: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeamRecord {
    pub team_id: String,
    pub object_permission: Option<Box<ObjectPermissionRecord>>,
    pub metadata: Option<Box<TeamPolicyMetadata>>,
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
    pub budget_limits: Vec<BudgetWindow>,
    pub model_max_budget: BTreeMap<String, ModelBudget>,
    pub model_spend: BTreeMap<String, f64>,
    pub model_id: Option<i64>,
    pub litellm_model_table: Option<TeamModelAliasTable>,
    pub router_settings: Option<RouterSettings>,
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
    pub member_budget_extras: BudgetExtras,
}

/// A user's row in a team. Its limits and budget are its budget row's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MembershipRecord {
    pub user_id: String,
    pub team_id: String,
    pub limits: MinuteLimits,
    pub budget: Budget,
    pub rollover_credit: Option<f64>,
    pub soft_budget: Option<f64>,
    pub tpd_limit: Option<i64>,
    pub max_parallel_requests: Option<i64>,
    pub budget_duration: Option<String>,
    pub budget_extras: BudgetExtras,
}

/// An organization. Its limits and budget are its budget row's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OrgRecord {
    pub organization_id: String,
    pub object_permission: Option<Box<ObjectPermissionRecord>>,
    pub metadata: Option<Box<OrganizationMetadata>>,
    pub organization_alias: Option<String>,
    pub models: Vec<String>,
    pub limits: MinuteLimits,
    pub budget: Budget,
    pub soft_budget: Option<f64>,
    pub tpd_limit: Option<i64>,
    pub max_parallel_requests: Option<i64>,
    pub budget_duration: Option<String>,
    pub model_spend: BTreeMap<String, f64>,
    pub budget_extras: BudgetExtras,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProjectRecord {
    pub project_id: String,
    pub object_permission: Option<Box<ObjectPermissionRecord>>,
    pub metadata: Option<Box<ProjectMetadata>>,
    pub project_alias: Option<String>,
    pub team_id: Option<String>,
    pub models: Vec<String>,
    pub blocked: bool,
    pub limits: MinuteLimits,
    pub budget: Budget,
    pub soft_budget: Option<f64>,
    pub tpd_limit: Option<i64>,
    pub max_parallel_requests: Option<i64>,
    pub budget_duration: Option<String>,
    pub model_spend: BTreeMap<String, f64>,
    pub model_rpm_limit: BTreeMap<String, i64>,
    pub model_tpm_limit: BTreeMap<String, i64>,
    pub budget_extras: BudgetExtras,
}

/// The request's end user. `found` is whether it has a row, so a request that names none, or one
/// with no row, reads an empty record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EndUserRecord {
    pub end_user_id: Option<String>,
    pub object_permission: Option<Box<ObjectPermissionRecord>>,
    pub found: bool,
    pub blocked: bool,
    pub default_model: Option<String>,
    pub models: Vec<String>,
    pub limits: MinuteLimits,
    pub budget: Budget,
    pub soft_budget: Option<f64>,
    pub tpd_limit: Option<i64>,
    pub max_parallel_requests: Option<i64>,
    pub budget_duration: Option<String>,
    pub allowed_model_region: Option<ModelRegion>,
    pub budget_extras: BudgetExtras,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BudgetExtras {
    pub model_max_budget: BTreeMap<String, ModelBudget>,
    pub model_rpm_limit: BTreeMap<String, i64>,
    pub model_tpm_limit: BTreeMap<String, i64>,
    pub allowed_models: Vec<String>,
    pub rollover_max_budget: Option<f64>,
    pub temp_budget_increase: Option<f64>,
    pub temp_budget_expiry: Option<Timestamp>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObjectPermissionRecord {
    pub object_permission_id: String,
    pub mcp_servers: Vec<String>,
    pub mcp_access_groups: Vec<String>,
    pub mcp_tool_permissions: BTreeMap<String, Vec<String>>,
    pub vector_stores: Vec<String>,
    pub agents: Vec<String>,
    pub agent_access_groups: Vec<String>,
    pub models: Vec<String>,
    pub mcp_toolsets: Vec<String>,
    pub blocked_tools: Vec<String>,
    pub search_tools: Vec<String>,
    pub mcp_tool_search_enabled: Option<bool>,
    pub skills: Vec<String>,
    pub apps: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct TeamPolicyMetadata {
    pub guardrails: Option<Vec<String>>,
    pub policies: Option<Vec<String>>,
    pub disable_global_guardrails: Option<bool>,
    pub opted_out_global_guardrails: Option<Vec<String>>,
    pub logging: Option<Vec<StoredLoggingCallback>>,
    pub callback_settings: Option<LegacyCallbackSettings>,
    pub model_config: Option<ModelCredentialConfig>,
    pub model_tpm_limit_counts_cached_tokens: Option<BTreeMap<String, bool>>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrganizationMetadata {
    pub model_rpm_limit: Option<BTreeMap<String, i64>>,
    pub model_tpm_limit: Option<BTreeMap<String, i64>>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectMetadata {
    pub guardrails: Option<Vec<String>>,
    pub policies: Option<Vec<String>>,
    pub model_rpm_limit: Option<BTreeMap<String, i64>>,
    pub model_tpm_limit: Option<BTreeMap<String, i64>>,
    pub model_itpm_limit: Option<BTreeMap<String, i64>>,
    pub model_otpm_limit: Option<BTreeMap<String, i64>>,
    pub model_config: Option<ModelCredentialConfig>,
}

pub type ModelCredentialConfig = BTreeMap<String, ProviderCredentials>;

#[serde_as]
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(transparent)]
pub struct ProviderCredentials(
    #[serde_as(as = "Map<_, _>")] pub Vec<(String, CredentialReference)>,
);

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CredentialReference {
    pub litellm_credentials: Option<SecretValue>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredLoggingCallback {
    pub callback_name: String,
    pub callback_type: Option<LoggingCallbackType>,
    pub callback_vars: BTreeMap<String, SecretValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, strum::VariantArray)]
pub enum LoggingCallbackType {
    #[serde(rename = "success")]
    Success,
    #[serde(rename = "failure")]
    Failure,
    #[serde(rename = "success_and_failure")]
    SuccessAndFailure,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyCallbackSettings {
    pub success_callback: Option<Vec<String>>,
    pub failure_callback: Option<Vec<String>>,
    pub callbacks: Option<Vec<String>>,
    pub callback_vars: Option<BTreeMap<String, SecretValue>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BudgetWindow {
    pub budget_duration: String,
    pub max_budget: f64,
    pub reset_at: Option<Timestamp>,
    pub window_start: Option<Timestamp>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ModelBudget {
    Amount(f64),
    Config(ModelBudgetConfig),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelBudgetConfig {
    #[serde(alias = "budget_limit")]
    pub max_budget: Option<f64>,
    #[serde(alias = "time_period")]
    pub budget_duration: Option<String>,
    pub rpm_limit: Option<i64>,
    pub tpm_limit: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TeamModelAliasTable {
    pub id: Option<i64>,
    pub model_aliases: BTreeMap<String, String>,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr, strum::VariantArray,
)]
pub enum ModelRegion {
    #[strum(serialize = "eu")]
    Eu,
    #[strum(serialize = "us")]
    Us,
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
