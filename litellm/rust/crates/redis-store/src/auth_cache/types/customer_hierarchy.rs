use litellm_python_compat::datetime::IsoDateTime;

use super::{
    BudgetLimitEntry, ExcludedIfNone, Ints,
    record::{Literal, record, required},
};

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr, strum::VariantArray,
)]
pub enum CustomerScopeKey {
    #[strum(serialize = "customer_org")]
    Organization,
    #[strum(serialize = "customer_team")]
    Team,
    #[strum(serialize = "customer_member")]
    Member,
}

impl Literal for CustomerScopeKey {}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr, strum::VariantArray,
)]
pub enum CustomerCapPolicy {
    #[strum(serialize = "pause_until_reset")]
    PauseUntilReset,
    #[strum(serialize = "pause_allow_increase")]
    PauseAllowIncrease,
}

impl Literal for CustomerCapPolicy {}

record! {
    /// One scope in the customer identity cache, as `CustomerScopeAuth` dumps it.
    pub struct CustomerScopeEntry {
        key: CustomerScopeKey = required!("key"),
        entity_id: String = required!("entity_id"),
        alias: Option<String>,
        spend: f64,
        budget_reset_at: Option<IsoDateTime>,
        max_budget: Option<f64>,
        budget_limits: Vec<BudgetLimitEntry>,
        soft_budget_duration: Option<String>,
        tpm_limit: Option<i64>,
        rpm_limit: Option<i64>,
        allowed_models: Option<Vec<String>>,
        model_tpm_limit: Option<Ints>,
        model_rpm_limit: Option<Ints>,
        cap_policy: Option<CustomerCapPolicy>,
    }
}

record! {
    /// The `CustomerHierarchyAuth` snapshot cached by member id or organization alias.
    pub struct CustomerHierarchyEntry {
        customer_org_id: Option<String>,
        customer_org_alias: Option<String>,
        customer_team_id: Option<String>,
        customer_team_alias: Option<String>,
        customer_member_id: Option<String>,
        customer_member_alias: Option<String>,
        app_id: ExcludedIfNone<String>,
        app_group_id: ExcludedIfNone<String>,
        app_alias: ExcludedIfNone<String>,
        app_customer_margin: ExcludedIfNone<f64>,
        scopes: Vec<CustomerScopeEntry> = required!("scopes"),
        team_scopes: Vec<CustomerScopeEntry>,
    }
}
