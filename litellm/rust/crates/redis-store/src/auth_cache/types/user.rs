use litellm_python_compat::datetime::IsoDateTime;

use super::{
    bag::Bag,
    budget_limit::BudgetLimitEntry,
    object_permission::ObjectPermissionEntry,
    organization_membership::OrganizationMembershipEntry,
    record::{ExcludedIfNone, record, required},
};

record! {
    /// `LiteLLM_UserTable.model_dump(mode="json")`, the value under
    /// [`AuthObject::User`](crate::auth_cache::AuthObject::User) and nested in an
    /// [`OrganizationEntry`](crate::auth_cache::OrganizationEntry)'s `users`.
    pub struct UserEntry {
        user_id: String = required!("user_id"),
        user_alias: Option<String>,
        team_id: Option<String>,
        sso_user_id: Option<String>,
        organization_id: Option<String>,
        object_permission_id: Option<String>,
        password_reset_required: Option<bool>,
        last_breach_check_at: Option<IsoDateTime>,
        teams: Vec<String>,
        user_role: Option<String>,
        max_budget: Option<f64>,
        rollover_max_budget: ExcludedIfNone<f64>,
        rollover_credit: ExcludedIfNone<f64>,
        budget_limits: Option<Vec<BudgetLimitEntry>>,
        spend: f64,
        user_email: Option<String>,
        models: Vec<String>,
        metadata: Option<Bag>,
        max_parallel_requests: Option<i64>,
        tpm_limit: Option<i64>,
        rpm_limit: Option<i64>,
        budget_duration: Option<String>,
        budget_reset_at: Option<IsoDateTime>,
        allowed_cache_controls: Vec<String>,
        policies: Vec<String>,
        model_spend: Option<Bag> = Some(Bag::default()),
        model_max_budget: Option<Bag> = Some(Bag::default()),
        created_at: Option<IsoDateTime>,
        updated_at: Option<IsoDateTime>,
        organization_memberships: Option<Vec<OrganizationMembershipEntry>>,
        object_permission: Option<ObjectPermissionEntry>,
    }
}
