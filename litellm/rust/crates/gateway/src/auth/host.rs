use std::time::Duration;

use litellm_config::LitellmRole;
use litellm_gateway_auth_core::{
    AgentIdentityFailure, AuthHost, Error, KeyHash, OperatorHooks, Rows, Token,
    authn::{
        jwt::{KeyMappingRow, KeyMappingWrite, MappedKey},
        key::{CachedTeam, Keyed},
    },
    identity::{
        jwt::{
            AliasRead, DelegatedTeam, FinishedJwt, IdentityRead, ManagedAgent, ModelAccess, Org,
            RegisteredAgent, TeamRead, UserLookup, UserRead,
        },
        key::Owner,
    },
};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NoRows;

impl Rows for NoRows {
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

pub(crate) enum NoOperatorCode {}

impl OperatorHooks for NoOperatorCode {
    async fn custom_validate(&self, _: &Value) -> Result<bool, Error> {
        match *self {}
    }
}

// TODO: the gateway has no database, key cache or HTTP client behind its host yet, so the cache
// reads find nothing and every other step fails. Only the master key authenticates until then
/// The gateway's `AuthHost`.
pub(crate) struct Host;

fn unanswered<T>() -> Result<T, Error> {
    Err(Error::HostFailed)
}

impl AuthHost for Host {
    type Rows = NoRows;

    async fn cached_key(&self, _: &KeyHash) -> Result<Option<Keyed<()>>, Error> {
        Ok(None)
    }
    async fn cached_team(&self, _: &str) -> Result<Option<CachedTeam<()>>, Error> {
        Ok(None)
    }
    async fn prefetch_identity(&self, _: Option<&Token>, _: Option<&str>) -> Result<(), Error> {
        Ok(())
    }
    async fn evict_key(&self, _: &KeyHash) -> Result<(), Error> {
        Ok(())
    }
    async fn end_user(&self, end_user_id: Option<&str>) -> Result<(), Error> {
        match end_user_id {
            None => Ok(()),
            Some(_) => unanswered(),
        }
    }
    fn operator_hooks(&self) -> Option<&impl OperatorHooks> {
        None::<&NoOperatorCode>
    }

    async fn db_key(&self, _: &KeyHash, _: &()) -> Result<Option<Keyed<()>>, Error> {
        unanswered()
    }
    async fn key_by_hash(&self, _: &KeyHash) -> Result<Option<MappedKey<()>>, Error> {
        unanswered()
    }
    async fn key_mapping_hash(
        &self,
        _: &str,
        _: &str,
        _: Option<&str>,
    ) -> Result<Option<KeyHash>, Error> {
        unanswered()
    }
    async fn reusable_key_hash(&self, _: &str, _: Option<&str>) -> Result<Option<KeyHash>, Error> {
        unanswered()
    }
    async fn key_owner(&self, _: &str) -> Result<Option<Owner<()>>, Error> {
        unanswered()
    }
    async fn user(&self, _: &UserLookup<'_>) -> Result<UserRead<()>, Error> {
        unanswered()
    }
    async fn resolve_user_identity(
        &self,
        _: &UserLookup<'_>,
        _: Option<&str>,
    ) -> Result<IdentityRead<NoRows>, Error> {
        unanswered()
    }
    async fn key_team(&self, _: &str) -> Result<Option<()>, Error> {
        unanswered()
    }
    async fn team(&self, _: &str, _: bool) -> Result<TeamRead<()>, Error> {
        unanswered()
    }
    async fn granting_team(&self, _: &str) -> Result<TeamRead<()>, Error> {
        unanswered()
    }
    async fn team_by_alias(&self, _: &str) -> Result<AliasRead<()>, Error> {
        unanswered()
    }
    async fn load_team_membership(&self, _: &str, _: &str) -> Result<Option<()>, Error> {
        unanswered()
    }
    async fn org(&self, _: &str) -> Result<Option<Org<()>>, Error> {
        unanswered()
    }
    async fn org_by_alias(&self, _: &str) -> Result<Option<Org<()>>, Error> {
        unanswered()
    }
    async fn jwt_end_user(&self, _: &str) -> Result<(), Error> {
        unanswered()
    }
    async fn object_permission(&self, _: &str) -> Result<Option<()>, Error> {
        unanswered()
    }
    async fn project(&self, _: &str) -> Result<Option<()>, Error> {
        unanswered()
    }
    async fn managed_agent(&self, _: &Value) -> Result<Option<ManagedAgent<()>>, Error> {
        unanswered()
    }
    async fn registered_agent(&self, _: &str) -> Result<Option<RegisteredAgent>, Error> {
        unanswered()
    }
    async fn prefetch_auth_objects(&self, _: &(), _: &()) -> Result<(), Error> {
        unanswered()
    }
    async fn set_user_role(&self, _: &(), _: LitellmRole) -> Result<(), Error> {
        unanswered()
    }
    async fn patch_team_membership(&self, _: &(), _: &[String], _: &[String]) -> Result<(), Error> {
        unanswered()
    }
    async fn add_team_member(&self, _: &str, _: &str) -> Result<(), Error> {
        unanswered()
    }
    async fn create_key_mapping(&self, _: &KeyMappingRow<'_>) -> Result<KeyMappingWrite, Error> {
        unanswered()
    }
    async fn record_agent_authentication(
        &self,
        _: &(),
    ) -> Result<Option<AgentIdentityFailure>, Error> {
        unanswered()
    }
    async fn fetch_json(&self, _: &str) -> Result<Option<Value>, Error> {
        unanswered()
    }
    async fn fetch_userinfo(&self, _: &str, _: &Token) -> Result<Option<Value>, Error> {
        unanswered()
    }
    async fn cached_json(&self, _: &str) -> Result<Option<Value>, Error> {
        unanswered()
    }
    async fn cache_json(&self, _: &str, _: &Value, _: Duration) -> Result<(), Error> {
        unanswered()
    }
    async fn forget_json(&self, _: &str) -> Result<(), Error> {
        unanswered()
    }
    async fn normalize_model(&self, _: &()) -> Result<(), Error> {
        unanswered()
    }
    async fn validated_end_user_id(&self, _: &str) -> Result<Option<String>, Error> {
        unanswered()
    }
    async fn key_skips_budget_checks(&self, _: &()) -> Result<bool, Error> {
        unanswered()
    }
    async fn team_model_access(&self, _: &(), _: &str) -> Result<ModelAccess, Error> {
        unanswered()
    }
    async fn delegated_agent_team(&self, _: &DelegatedTeam<'_>) -> Result<Option<String>, Error> {
        unanswered()
    }
    async fn team_member_budget(&self, _: &(), _: Option<&()>) -> Result<(), Error> {
        unanswered()
    }
    async fn key_budgets(&self, _: &(), _: Option<&()>, _: bool) -> Result<(), Error> {
        unanswered()
    }
    async fn proxy_budget_alert(&self, _: &()) -> Result<(), Error> {
        unanswered()
    }
    async fn finish_jwt(&self, _: FinishedJwt<'_, NoRows>) -> Result<(), Error> {
        unanswered()
    }
    async fn ui_key(&self, _: &Token) -> Result<Option<Keyed<()>>, Error> {
        unanswered()
    }
    async fn with_team(&self, _: &(), _: &()) -> Result<(), Error> {
        unanswered()
    }
    async fn with_temp_budget(&self, _: Keyed<()>) -> Result<Keyed<()>, Error> {
        unanswered()
    }
}
