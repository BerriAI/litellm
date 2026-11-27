//! A project as the database holds it, in the shape the auth cache holds it.

use std::sync::Arc;

use litellm_redis_store::auth_cache::ProjectEntry;
use litellm_repositories::{Project, ProjectRepository};

use crate::{
    Error,
    budget::budget_entry,
    column::{nullable, time},
    error::within,
};

/// The project as the database holds it, in the shape the auth cache holds it: the row with its
/// budget, and nothing else nested.
pub struct ProjectDatabase {
    projects: Arc<ProjectRepository>,
}

impl ProjectDatabase {
    pub fn new(projects: Arc<ProjectRepository>) -> Self {
        Self { projects }
    }

    /// The project `project_id`, or `None` when no row has that id. The object permission the
    /// project points at is not read, and `last_refreshed_at` is unset, since the host that caches
    /// the project stamps it.
    pub async fn project(&self, project_id: &str) -> Result<Option<ProjectEntry>, Error> {
        self.projects
            .find_with_budget(project_id)
            .await?
            .map(project_entry)
            .transpose()
    }
}

/// What `LiteLLM_ProjectTableCachedObj.model_dump(mode="json")` holds for `project`, or the
/// column the model would refuse the row for.
fn project_entry(project: Project) -> Result<ProjectEntry, Error> {
    let Project {
        project_id,
        project_alias,
        description,
        team_id,
        budget_id,
        metadata,
        models,
        spend,
        model_spend,
        model_rpm_limit,
        model_tpm_limit,
        blocked,
        object_permission_id,
        created_at,
        created_by,
        updated_at,
        updated_by,
        litellm_budget_table,
    } = project;
    Ok(ProjectEntry {
        project_id,
        project_alias,
        description,
        team_id,
        budget_id,
        metadata: nullable("metadata", Some(&metadata))?,
        models,
        spend,
        model_spend: nullable("model_spend", Some(&model_spend))?,
        model_rpm_limit: nullable("model_rpm_limit", Some(&model_rpm_limit))?,
        model_tpm_limit: nullable("model_tpm_limit", Some(&model_tpm_limit))?,
        blocked,
        object_permission_id,
        created_by: Some(created_by),
        updated_by: Some(updated_by),
        created_at: time("created_at", Some(created_at))?,
        updated_at: time("updated_at", Some(updated_at))?,
        litellm_budget_table: litellm_budget_table
            .map(|budget| budget_entry(budget.0))
            .transpose()
            .map_err(within("litellm_budget_table"))?,
        object_permission: None,
        last_refreshed_at: None,
        unread: Vec::new(),
    })
}
