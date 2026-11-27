use litellm_python_compat::{Value, datetime::IsoDateTime};

use super::{
    bag::Bag,
    budget_limit::BudgetLimitEntry,
    member::MemberEntry,
    object_permission::ObjectPermissionEntry,
    record::{Field, record, required},
};

/// `LiteLLM_ModelTable.model_aliases`, which the model types `str | dict`.
#[derive(Clone, PartialEq, veil::Redact)]
pub enum ModelAliases {
    Text(#[redact(with = "[REDACTED]")] String),
    Map(Bag),
}

impl Field for ModelAliases {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::Str(text) => Some(Self::Text(text.clone())),
            value => Bag::read(value).map(Self::Map),
        }
    }

    fn write(self) -> Value {
        match self {
            Self::Text(text) => Value::Str(text),
            Self::Map(aliases) => aliases.write(),
        }
    }
}

record! {
    /// A team's model alias table, as `LiteLLM_ModelTable` dumps it nested in the team.
    pub struct ModelTableEntry {
        id: Option<i64>,
        model_aliases: Option<ModelAliases>,
        created_by: String = required!("created_by"),
        updated_by: String = required!("updated_by"),
        #[redact(with = "[REDACTED]")]
        team: Option<Value>,
    }
}

record! {
    /// `LiteLLM_TeamTableCachedObj.model_dump(mode="json")`, the value under
    /// [`AuthObject::Team`](crate::auth_cache::AuthObject::Team).
    pub struct TeamEntry {
        team_alias: Option<String>,
        team_id: String = required!("team_id"),
        organization_id: Option<String>,
        admins: Vec<String>,
        members: Vec<String>,
        members_with_roles: Vec<MemberEntry>,
        team_member_permissions: Option<Vec<String>>,
        metadata: Option<Bag>,
        tpm_limit: Option<i64>,
        rpm_limit: Option<i64>,
        tpd_limit: Option<i64>,
        max_budget: Option<f64>,
        soft_budget: Option<f64>,
        budget_duration: Option<String>,
        budget_limits: Option<Vec<BudgetLimitEntry>>,
        models: Vec<String>,
        blocked: bool,
        router_settings: Option<Bag>,
        access_group_ids: Option<Vec<String>>,
        default_team_member_models: Option<Vec<String>>,
        spend: Option<f64>,
        max_parallel_requests: Option<i64>,
        budget_reset_at: Option<IsoDateTime>,
        model_id: Option<i64>,
        model_spend: Option<Bag> = Some(Bag::default()),
        model_max_budget: Option<Bag> = Some(Bag::default()),
        policies: Option<Vec<String>>,
        allow_team_guardrail_config: Option<bool> = Some(false),
        litellm_model_table: Option<ModelTableEntry>,
        object_permission: Option<ObjectPermissionEntry>,
        object_permission_id: Option<String>,
        updated_at: Option<IsoDateTime>,
        created_at: Option<IsoDateTime>,
        last_refreshed_at: Option<f64>,
    }
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use rstest::rstest;

    use super::TeamEntry;
    use crate::{FromValue, auth_cache::Bag};

    #[rstest]
    fn a_team_entry_reads_flags_and_numbers_as_pydantic_does() {
        let team = |entry: &str| TeamEntry::from_value(&json::loads(entry).unwrap());
        let lax = team(r#"{"team_id": "t", "blocked": "yes", "rpm_limit": "3", "tpm_limit": 3.0, "spend": "1.5"}"#)
            .unwrap();

        assert_eq!(
            (
                (lax.blocked, lax.rpm_limit, lax.tpm_limit, lax.spend),
                team(r#"{"team_id": "t", "rpm_limit": 3.5}"#).is_err(),
                team(r#"{"team_id": "t", "blocked": "maybe"}"#).is_err(),
            ),
            ((true, Some(3), Some(3), Some(1.5)), true, true)
        );
    }

    #[rstest]
    fn a_team_entry_needs_its_id_and_takes_the_models_defaults_for_the_rest() {
        let team = TeamEntry::from_value(&json::loads(r#"{"team_id": "team"}"#).unwrap()).unwrap();

        assert_eq!(
            (
                team.team_id.as_str(),
                team.blocked,
                team.models.len(),
                team.model_spend,
                team.model_max_budget,
                team.allow_team_guardrail_config,
                TeamEntry::from_value(&json::loads("{}").unwrap()).is_err(),
            ),
            (
                "team",
                false,
                0,
                Some(Bag::default()),
                Some(Bag::default()),
                Some(false),
                true,
            )
        );
    }
}
