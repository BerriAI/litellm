use litellm_python_compat::Value;

use super::{
    budget::BudgetFullEntry,
    record::{ExcludedIfNone, record, required},
};
use crate::{DecodeError, FromValue, IntoValue};

/// `NO_TEAM_MEMBERSHIP_SENTINEL`.
const NOT_MEMBER: &str = "__no_team_membership__";

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

/// What `_fetch_team_membership_from_db` caches under
/// [`AuthObject::TeamMembershipReservation`](crate::auth_cache::AuthObject::TeamMembershipReservation):
/// the membership, or `NotMember` for a user who is not a member of the team.
#[derive(Clone, Debug, PartialEq)]
pub enum TeamMembershipReservation {
    Member(TeamMembershipEntry),
    NotMember,
}

impl FromValue for TeamMembershipReservation {
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        match value {
            Value::Str(word) if word == NOT_MEMBER => Ok(Self::NotMember),
            _ => TeamMembershipEntry::from_value(value).map(Self::Member),
        }
    }
}

impl IntoValue for TeamMembershipReservation {
    fn into_value(self) -> Value {
        match self {
            Self::Member(membership) => membership.into_value(),
            Self::NotMember => Value::Str(NOT_MEMBER.to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use rstest::rstest;

    use super::{TeamMembershipEntry, TeamMembershipReservation};
    use crate::{FromValue, IntoValue};

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

    #[rstest]
    fn a_user_who_is_not_a_member_reads_back_as_not_a_member() {
        let value = TeamMembershipReservation::NotMember.into_value();

        assert_eq!(
            TeamMembershipReservation::from_value(&value).ok(),
            Some(TeamMembershipReservation::NotMember)
        );
    }

    #[rstest]
    fn a_membership_reads_back_as_the_membership() {
        let entry = json::loads(r#"{"user_id": "u", "team_id": "t"}"#).unwrap();
        let membership = TeamMembershipEntry::from_value(&entry).unwrap();
        let value = TeamMembershipReservation::Member(membership.clone()).into_value();

        assert_eq!(
            TeamMembershipReservation::from_value(&value).ok(),
            Some(TeamMembershipReservation::Member(membership))
        );
    }

    #[rstest]
    fn other_text_is_no_membership_entry() {
        let value = json::loads(r#""__not_the_word__""#).unwrap();

        assert!(TeamMembershipReservation::from_value(&value).is_err());
    }
}
