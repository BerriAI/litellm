//! An organization as the database holds it, in the shape the auth cache holds it.

use std::sync::Arc;

use litellm_redis_store::auth_cache::OrganizationEntry;
use litellm_repositories::{Organization, OrganizationRepository};

use crate::{Error, budget::budget_entry, column::nullable, error::within};

/// The organization as the database holds it, in the shape the auth cache holds it: the row with
/// its budget when asked for, and nothing else nested.
pub struct OrganizationDatabase {
    organizations: Arc<OrganizationRepository>,
}

impl OrganizationDatabase {
    pub fn new(organizations: Arc<OrganizationRepository>) -> Self {
        Self { organizations }
    }

    /// The organization `organization_id`, or `None` when no row has that id. With
    /// `include_budget_table` it is the shape `get_org_object` caches under `org_id:{id}:with_budget`,
    /// and without it the shape under `org_id:{id}`, where `litellm_budget_table` is `None`. The
    /// object permission the organization points at is not read.
    pub async fn organization(
        &self,
        organization_id: &str,
        include_budget_table: bool,
    ) -> Result<Option<OrganizationEntry>, Error> {
        self.organizations
            .find_with_optional_budget(organization_id, include_budget_table)
            .await?
            .map(organization_entry)
            .transpose()
    }
}

/// What `LiteLLM_OrganizationTable.model_dump(mode="json")` holds for `organization`, or the
/// column the model would refuse the row for.
fn organization_entry(organization: Organization) -> Result<OrganizationEntry, Error> {
    let Organization {
        organization_id,
        organization_alias,
        budget_id,
        metadata,
        models,
        spend,
        model_spend,
        object_permission_id,
        created_by,
        updated_by,
        litellm_budget_table,
    } = organization;
    Ok(OrganizationEntry {
        organization_id: Some(organization_id),
        organization_alias: Some(organization_alias),
        budget_id,
        spend,
        metadata: nullable("metadata", Some(&metadata))?,
        models,
        model_spend: nullable("model_spend", Some(&model_spend))?,
        created_by,
        updated_by,
        users: None,
        litellm_budget_table: litellm_budget_table
            .map(|budget| budget_entry(budget.0))
            .transpose()
            .map_err(within("litellm_budget_table"))?,
        object_permission: None,
        object_permission_id,
        unread: Vec::new(),
    })
}
