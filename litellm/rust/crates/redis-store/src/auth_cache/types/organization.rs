use litellm_python_compat::Value;

use super::{
    bag::Bag,
    budget::BudgetEntry,
    object_permission::ObjectPermissionEntry,
    record::{record, required},
};

record! {
    /// `LiteLLM_OrganizationTable.model_dump(mode="json")`, the value `get_org_object` caches under
    /// [`AuthObject::OrgWithBudget`](crate::auth_cache::AuthObject::OrgWithBudget) with its budget
    /// and under [`AuthObject::Org`](crate::auth_cache::AuthObject::Org) and the alias keys without.
    pub struct OrganizationEntry {
        organization_id: Option<String>,
        organization_alias: Option<String>,
        budget_id: String = required!("budget_id"),
        spend: f64,
        metadata: Option<Bag>,
        models: Vec<String>,
        model_spend: Option<Bag> = Some(Bag::default()),
        created_by: String = required!("created_by"),
        updated_by: String = required!("updated_by"),
        #[redact(with = "[REDACTED]")]
        users: Option<Vec<Value>>,
        litellm_budget_table: Option<BudgetEntry>,
        object_permission: Option<ObjectPermissionEntry>,
        object_permission_id: Option<String>,
    }
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use rstest::rstest;

    use super::OrganizationEntry;
    use crate::{FromValue, IntoValue};

    #[rstest]
    fn an_organization_entry_takes_the_models_defaults_for_what_it_lacks() {
        let entry = OrganizationEntry::from_value(
            &json::loads(
                r#"{"budget_id": "budget", "created_by": "creator", "updated_by": "creator"}"#,
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(
            json::dumps(&entry.into_value()).unwrap(),
            concat!(
                r#"{"organization_id": null, "organization_alias": null, "budget_id": "budget","#,
                r#" "spend": 0.0, "metadata": null, "models": [], "model_spend": {},"#,
                r#" "created_by": "creator", "updated_by": "creator", "users": null,"#,
                r#" "litellm_budget_table": null, "object_permission": null,"#,
                r#" "object_permission_id": null}"#,
            )
        );
    }
}
