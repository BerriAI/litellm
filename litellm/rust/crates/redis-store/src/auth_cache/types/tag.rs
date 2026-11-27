use litellm_python_compat::datetime::IsoDateTime;

use super::{
    bag::Bag,
    budget::BudgetEntry,
    record::{record, required},
};

record! {
    /// `LiteLLM_TagTable.model_dump(mode="json")`, the value under
    /// [`AuthObject::Tag`](crate::auth_cache::AuthObject::Tag).
    pub struct TagEntry {
        tag_name: String = required!("tag_name"),
        description: Option<String>,
        models: Vec<String> => default_on_null,
        model_info: Option<Bag>,
        spend: f64 => default_on_null,
        budget_id: Option<String>,
        litellm_budget_table: Option<BudgetEntry>,
        created_at: Option<IsoDateTime>,
        created_by: Option<String>,
        updated_at: Option<IsoDateTime>,
    }
}
