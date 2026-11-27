use super::{
    bag::Bag,
    budget::BudgetEntry,
    object_permission::ObjectPermissionEntry,
    record::{record, required},
    user::UserEntry,
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
        users: Option<Vec<UserEntry>>,
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

    const ORGANIZATION: &str =
        r#""budget_id": "budget", "created_by": "creator", "updated_by": "creator""#;

    fn entry(fields: &str) -> Result<OrganizationEntry, crate::DecodeError> {
        OrganizationEntry::from_value(&json::loads(&format!("{{{ORGANIZATION}{fields}}}")).unwrap())
    }

    #[rstest]
    fn an_organization_entry_takes_the_models_defaults_for_what_it_lacks() {
        assert_eq!(
            json::dumps(&entry("").unwrap().into_value()).unwrap(),
            concat!(
                r#"{"organization_id": null, "organization_alias": null, "budget_id": "budget","#,
                r#" "spend": 0.0, "metadata": null, "models": [], "model_spend": {},"#,
                r#" "created_by": "creator", "updated_by": "creator", "users": null,"#,
                r#" "litellm_budget_table": null, "object_permission": null,"#,
                r#" "object_permission_id": null}"#,
            )
        );
    }

    #[rstest]
    fn an_organization_entry_takes_users_only_as_user_entries() {
        assert_eq!(
            (
                entry(r#", "users": [{"user_id": "member"}]"#)
                    .unwrap()
                    .users
                    .unwrap()
                    .into_iter()
                    .map(|user| (user.user_id, user.spend))
                    .collect::<Vec<_>>(),
                entry(r#", "users": [{"user_email": "member@example.com"}]"#).is_err(),
            ),
            (vec![("member".to_owned(), 0.0)], true)
        );
    }
}
