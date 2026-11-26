use super::{
    budget::BudgetEntry,
    object_permission::ObjectPermissionEntry,
    record::{Literal, record, required},
};

/// The two regions `LiteLLM_EndUserTable.allowed_model_region` takes.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr, strum::VariantArray,
)]
#[strum(serialize_all = "lowercase")]
pub enum ModelRegion {
    Eu,
    Us,
}

impl Literal for ModelRegion {}

record! {
    /// `LiteLLM_EndUserTable.model_dump(mode="json")`, the value `get_end_user_object` caches
    /// under [`AuthObject::EndUser`](crate::auth_cache::AuthObject::EndUser).
    pub struct EndUserEntry {
        user_id: String = required!("user_id"),
        blocked: bool = required!("blocked"),
        alias: Option<String>,
        spend: f64,
        allowed_model_region: Option<ModelRegion>,
        default_model: Option<String>,
        models: Vec<String>,
        budget_id: Option<String>,
        litellm_budget_table: Option<BudgetEntry>,
        object_permission_id: Option<String>,
        object_permission: Option<ObjectPermissionEntry>,
    }
}
