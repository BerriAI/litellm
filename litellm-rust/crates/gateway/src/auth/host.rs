use std::time::Duration;

use litellm_config::LitellmRole;
use litellm_gateway_auth_core::{
    AuthHost, Error,
    authorize::Membership,
    key_budget::KeyBudgetAlert,
    budgets::{BudgetAlert, MemberBudget, NamedBudget, RecordedBudget, SpendCounter},
    jwt::{
        AgentIdentityFailure, AliasRead, DelegatedTeam, FinishedJwt, IdentityRead, KeyMappingRow,
        KeyMappingWrite, ManagedAgent, MappedKey, ModelAccess, Org, RegisteredAgent, TeamRead,
        UserLookup, UserRead,
    },
    key::{CachedTeam, Keyed, Owner},
};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NoRows;

impl litellm_gateway_auth_core::Rows for NoRows {
    type Key = ();
    type User = ();
    type Team = ();
    type EndUser = ();
    type ObjectPermission = ();
    type Project = ();
    type Membership = ();
    type Org = ();
    type Agent = ();
}

// TODO: the key path's reads find nothing and every other step fails until the gateway has a
// database, the key cache and an HTTP client to answer them, so only the master key authenticates
/// The gateway's `AuthHost`, with no database or cache behind it.
pub(crate) struct Host;

impl AuthHost for Host {
    type Rows = NoRows;

    async fn cached_key(&self, _: &str) -> Result<Option<Keyed<()>>, Error> {
        Ok(None)
    }
    async fn ui_key(&self, _: &str) -> Result<Option<Keyed<()>>, Error> {
        Ok(None)
    }
    async fn db_key(&self, _: &str, _: &()) -> Result<Option<Keyed<()>>, Error> {
        Ok(None)
    }
    async fn key_owner(&self, _: &str) -> Result<Option<Owner<()>>, Error> {
        Ok(None)
    }
    async fn cached_team(&self, _: &str) -> Result<Option<CachedTeam<()>>, Error> {
        Ok(None)
    }
    async fn key_team(&self, _: &str) -> Result<Option<()>, Error> {
        Ok(None)
    }
    async fn object_permission(&self, _: &str) -> Result<Option<()>, Error> {
        Ok(None)
    }
    async fn project(&self, _: &str) -> Result<Option<()>, Error> {
        Ok(None)
    }
    async fn with_session_grants(
        &self,
        _: &(),
        _: &(),
        _: Option<&()>,
        _: Option<&()>,
    ) -> Result<Keyed<()>, Error> {
        Err(Error::HostFailed)
    }
    async fn with_team(&self, _: &(), _: &()) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn with_temp_budget(&self, _: &()) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn normalize_model(&self, _: &()) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn prefetch_auth_objects(&self, _: &(), _: &()) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn evict_key(&self, _: &str) -> Result<(), Error> {
        Ok(())
    }
    async fn fetch_json(&self, _: &str) -> Result<Option<Value>, Error> {
        Err(Error::HostFailed)
    }
    async fn fetch_userinfo(&self, _: &str, _: &str) -> Result<Option<Value>, Error> {
        Err(Error::HostFailed)
    }
    async fn cached_json(&self, _: &str) -> Result<Option<Value>, Error> {
        Err(Error::HostFailed)
    }
    async fn cache_json(&self, _: &str, _: &Value, _: Duration) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn forget_json(&self, _: &str) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn team(&self, _: &str, _: bool) -> Result<TeamRead<()>, Error> {
        Err(Error::HostFailed)
    }
    async fn team_by_alias(&self, _: &str) -> Result<AliasRead<()>, Error> {
        Err(Error::HostFailed)
    }
    async fn granting_team(&self, _: &str) -> Result<TeamRead<()>, Error> {
        Err(Error::HostFailed)
    }
    async fn load_team_membership(&self, _: &str, _: &str) -> Result<Option<()>, Error> {
        Err(Error::HostFailed)
    }
    async fn user(&self, _: &UserLookup<'_>) -> Result<UserRead<()>, Error> {
        Err(Error::HostFailed)
    }
    async fn resolve_user_identity(
        &self,
        _: &UserLookup<'_>,
        _: Option<&str>,
    ) -> Result<IdentityRead<NoRows>, Error> {
        Err(Error::HostFailed)
    }
    async fn org(&self, _: &str) -> Result<Option<Org<()>>, Error> {
        Err(Error::HostFailed)
    }
    async fn org_by_alias(&self, _: &str) -> Result<Option<Org<()>>, Error> {
        Err(Error::HostFailed)
    }
    async fn jwt_end_user(&self, _: &str) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn set_user_role(&self, _: &(), _: LitellmRole) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn patch_team_membership(
        &self,
        _: &(),
        _: &[String],
        _: &[String],
    ) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn add_team_member(&self, _: &str, _: &str) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn key_mapping_hash(
        &self,
        _: &str,
        _: &str,
        _: Option<&str>,
    ) -> Result<Option<String>, Error> {
        Err(Error::HostFailed)
    }
    async fn key_by_hash(&self, _: &str) -> Result<Option<MappedKey<()>>, Error> {
        Err(Error::HostFailed)
    }
    async fn reusable_key_hash(&self, _: &str, _: Option<&str>) -> Result<Option<String>, Error> {
        Err(Error::HostFailed)
    }
    async fn create_key_mapping(&self, _: &KeyMappingRow<'_>) -> Result<KeyMappingWrite, Error> {
        Err(Error::HostFailed)
    }
    async fn managed_agent(&self, _: &Value) -> Result<Option<ManagedAgent<()>>, Error> {
        Err(Error::HostFailed)
    }
    async fn registered_agent(&self, _: &str) -> Result<Option<RegisteredAgent>, Error> {
        Err(Error::HostFailed)
    }
    async fn record_agent_authentication(
        &self,
        _: &(),
    ) -> Result<Option<AgentIdentityFailure>, Error> {
        Err(Error::HostFailed)
    }
    async fn unpriced_models(&self) -> Result<Vec<String>, Error> {
        Err(Error::HostFailed)
    }
    async fn agent_requires_trace_id(&self, _: &str) -> Result<bool, Error> {
        Err(Error::HostFailed)
    }
    async fn has_trace_id(&self, _: bool) -> Result<bool, Error> {
        Err(Error::HostFailed)
    }
    async fn team_organization(&self, _: &str) -> Result<Option<String>, Error> {
        Err(Error::HostFailed)
    }
    async fn merge_budget_tags(&self) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn check_team_model_access(&self, _: Membership<'_, ()>) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn check_model_grants(&self) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn check_project(&self) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn stamp_access_groups(&self, _: Membership<'_, ()>) -> Result<Vec<String>, Error> {
        Err(Error::HostFailed)
    }
    async fn spend(&self, _: SpendCounter<'_>) -> Result<f64, Error> {
        Err(Error::HostFailed)
    }
    async fn organization_budget(&self, _: &str) -> Result<Option<RecordedBudget>, Error> {
        Err(Error::HostFailed)
    }
    async fn tag_budgets(&self) -> Result<Vec<NamedBudget>, Error> {
        Err(Error::HostFailed)
    }
    async fn model_access_group_budgets(&self, _: &[String]) -> Result<Vec<NamedBudget>, Error> {
        Err(Error::HostFailed)
    }
    async fn team_member_budget_state(&self, _: Option<&()>) -> Result<MemberBudget, Error> {
        Err(Error::HostFailed)
    }
    async fn budget_alert(&self, _: BudgetAlert<'_>) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn check_guardrail_modification(&self) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn check_object_access(&self) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn custom_validate(&self, _: &Value) -> Result<bool, Error> {
        Err(Error::HostFailed)
    }
    async fn end_user(&self, _: &str, _: Option<&()>) -> Result<(), Error> {
        Ok(())
    }
    async fn key_skips_budget_checks(&self, _: &()) -> Result<bool, Error> {
        Err(Error::HostFailed)
    }
    async fn team_member_max_budget(&self, _: &str, _: &str) -> Result<Option<f64>, Error> {
        Err(Error::HostFailed)
    }
    async fn key_model_budgets(&self, _: &(), _: Option<&()>) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn throttle_key(&self, _: &(), _: f64) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn global_proxy_spend(&self) -> Result<Option<f64>, Error> {
        Err(Error::HostFailed)
    }
    async fn key_budget_alert(&self, _: &(), _: Option<&()>, _: KeyBudgetAlert) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn finish_jwt(&self, _: FinishedJwt<'_, NoRows>) -> Result<(), Error> {
        Err(Error::HostFailed)
    }
    async fn team_model_access(&self, _: &(), _: &str) -> Result<ModelAccess, Error> {
        Err(Error::HostFailed)
    }
    async fn delegated_agent_team(&self, _: &DelegatedTeam<'_>) -> Result<Option<String>, Error> {
        Err(Error::HostFailed)
    }
}
