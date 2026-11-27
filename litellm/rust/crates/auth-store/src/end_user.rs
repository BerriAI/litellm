//! An end user as the database holds it, in the shape the auth cache holds it.

use std::sync::Arc;

use litellm_redis_store::auth_cache::EndUserEntry;
use litellm_repositories::{EndUserRecord, EndUserRepository};

use crate::{
    Error, budget::budget_entry, error::within, object_permission::object_permission_entry,
};

/// The end user as the database holds it, in the shape the auth cache holds it: the row with its
/// budget and object permission.
pub struct EndUserDatabase {
    end_users: Arc<EndUserRepository>,
}

impl EndUserDatabase {
    pub fn new(end_users: Arc<EndUserRepository>) -> Self {
        Self { end_users }
    }

    /// The end user `end_user_id`, or `None` when no row has that id.
    pub async fn end_user(&self, end_user_id: &str) -> Result<Option<EndUserEntry>, Error> {
        self.end_users
            .find_with_budget_and_object_permission(end_user_id)
            .await?
            .map(end_user_entry)
            .transpose()
    }
}

/// What `LiteLLM_EndUserTable.model_dump(mode="json")` holds for `end_user`, or the column the
/// model would refuse the row for.
fn end_user_entry(end_user: EndUserRecord) -> Result<EndUserEntry, Error> {
    let EndUserRecord {
        user_id,
        alias,
        spend,
        allowed_model_region,
        default_model,
        models,
        budget_id,
        litellm_budget_table,
        object_permission_id,
        object_permission,
        blocked,
    } = end_user;
    Ok(EndUserEntry {
        user_id,
        blocked,
        alias,
        spend,
        allowed_model_region: allowed_model_region
            .map(|region| region.parse())
            .transpose()
            .map_err(|_| Error::row("allowed_model_region"))?,
        default_model,
        models,
        budget_id,
        litellm_budget_table: litellm_budget_table
            .map(|budget| budget_entry(budget.0))
            .transpose()
            .map_err(within("litellm_budget_table"))?,
        object_permission_id,
        object_permission: object_permission
            .map(|permission| object_permission_entry(permission.0))
            .transpose()
            .map_err(within("object_permission"))?,
        unread: Vec::new(),
    })
}
