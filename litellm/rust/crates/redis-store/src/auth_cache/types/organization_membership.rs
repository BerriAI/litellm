use litellm_python_compat::{Value, datetime::IsoDateTime};

use super::{
    budget::BudgetEntry,
    record::{record, required},
};

record! {
    /// `LiteLLM_OrganizationMembershipTable.model_dump(mode="json")`, nested in a
    /// [`UserEntry`](super::UserEntry). The model types `user` as any object, so it stays plain
    /// data.
    pub struct OrganizationMembershipEntry {
        user_id: String = required!("user_id"),
        organization_id: String = required!("organization_id"),
        user_role: Option<String>,
        spend: f64,
        budget_id: Option<String>,
        created_at: IsoDateTime = required!("created_at"),
        updated_at: IsoDateTime = required!("updated_at"),
        #[redact(with = "[REDACTED]")]
        user: Option<Value>,
        litellm_budget_table: Option<BudgetEntry>,
        user_email: Option<String>,
    }
}
