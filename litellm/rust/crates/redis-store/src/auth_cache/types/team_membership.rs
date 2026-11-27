use super::{
    budget::BudgetFullEntry,
    record::{ExcludedIfNone, record, required},
};

record! {
    /// `LiteLLM_TeamMembership.model_dump(mode="json")`, the value under
    /// [`AuthObject::TeamMembership`](crate::auth_cache::AuthObject::TeamMembership) and
    /// [`AuthObject::TeamMembershipReservation`](crate::auth_cache::AuthObject::TeamMembershipReservation),
    /// which holds Python's `"__no_team_membership__"` string instead for a user who is not a member.
    pub struct TeamMembershipEntry {
        user_id: String = required!("user_id"),
        team_id: String = required!("team_id"),
        budget_id: Option<String>,
        spend: Option<f64> = Some(0.0),
        total_spend: Option<f64> = Some(0.0),
        rollover_credit: ExcludedIfNone<f64>,
        litellm_budget_table: Option<BudgetFullEntry>,
    }
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use rstest::rstest;

    use super::TeamMembershipEntry;
    use crate::FromValue;

    #[rstest]
    #[case::a_full_budget(r#"{"budget_id": "b", "created_at": "2026-01-01T00:00:00Z"}"#, true)]
    #[case::a_budget_without_its_creation_time(r#"{"budget_id": "b"}"#, false)]
    fn a_membership_entry_takes_only_a_full_budget(#[case] budget: &str, #[case] decodes: bool) {
        let entry =
            format!(r#"{{"user_id": "u", "team_id": "t", "litellm_budget_table": {budget}}}"#);

        assert_eq!(
            TeamMembershipEntry::from_value(&json::loads(&entry).unwrap()).is_ok(),
            decodes
        );
    }
}
