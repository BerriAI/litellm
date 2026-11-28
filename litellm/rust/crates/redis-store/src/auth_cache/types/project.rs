use litellm_python_compat::datetime::IsoDateTime;

use super::{
    bag::Bag,
    budget::BudgetEntry,
    object_permission::ObjectPermissionEntry,
    record::{record, required},
};

record! {
    /// `LiteLLM_ProjectTableCachedObj.model_dump(mode="json")`, the value under
    /// [`AuthObject::Project`](crate::auth_cache::AuthObject::Project).
    pub struct ProjectEntry {
        project_id: String = required!("project_id"),
        project_alias: Option<String>,
        description: Option<String>,
        team_id: Option<String>,
        budget_id: Option<String>,
        metadata: Option<Bag>,
        models: Vec<String>,
        spend: f64,
        model_spend: Option<Bag>,
        model_rpm_limit: Option<Bag>,
        model_tpm_limit: Option<Bag>,
        blocked: bool,
        object_permission_id: Option<String>,
        created_by: Option<String>,
        updated_by: Option<String>,
        created_at: Option<IsoDateTime>,
        updated_at: Option<IsoDateTime>,
        litellm_budget_table: Option<BudgetEntry>,
        object_permission: Option<ObjectPermissionEntry>,
        last_refreshed_at: Option<f64>,
    }
}
