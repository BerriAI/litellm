use litellm_python_compat::Value;

use super::record::{Field, record, required};

/// The two roles `Member` takes.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr, strum::VariantArray,
)]
#[strum(serialize_all = "lowercase")]
pub enum MemberRole {
    Admin,
    User,
}

impl Field for MemberRole {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::Str(role) => role.parse().ok(),
            _ => None,
        }
    }

    fn write(self) -> Value {
        Value::from(<&'static str>::from(self))
    }
}

record! {
    /// A team member as `Member` dumps it, in a team's roster and nested in a key's entry.
    pub struct MemberEntry {
        user_id: Option<String>,
        user_email: Option<String>,
        role: MemberRole = required!("role"),
    }
    refuse_unless names_a_user;
}

fn names_a_user(member: &MemberEntry) -> bool {
    member.user_id.is_some() || member.user_email.is_some()
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use rstest::rstest;

    use super::MemberEntry;
    use crate::{DecodeError, FromValue, IntoValue, auth_cache::ObjectPermissionEntry};

    #[rstest]
    #[case::neither_id(r#"{"role": "user"}"#, "MemberEntry")]
    #[case::null_ids(
        r#"{"user_id": null, "user_email": null, "role": "user"}"#,
        "MemberEntry"
    )]
    #[case::a_role_the_model_does_not_take(r#"{"user_id": "u", "role": "owner"}"#, "role")]
    fn a_member_the_model_refuses_does_not_decode(#[case] entry: &str, #[case] refused: &str) {
        assert!(matches!(
            MemberEntry::from_value(&json::loads(entry).unwrap()),
            Err(DecodeError::Shape(field)) if field == refused
        ));
    }

    #[rstest]
    fn a_member_and_a_permission_keep_what_they_do_not_name() {
        let member = MemberEntry::from_value(
            &json::loads(r#"{"user_id": "u", "role": "user", "joined": "2026"}"#).unwrap(),
        )
        .unwrap();
        let permission = ObjectPermissionEntry::from_value(
            &json::loads(r#"{"object_permission_id": "p", "later": [1]}"#).unwrap(),
        )
        .unwrap();

        assert_eq!(
            (
                json::dumps(&member.into_value()).unwrap().as_str(),
                json::dumps(&permission.into_value())
                    .unwrap()
                    .ends_with(r#""skills": null, "apps": null, "later": [1]}"#),
            ),
            (
                r#"{"user_id": "u", "user_email": null, "role": "user", "joined": "2026"}"#,
                true
            )
        );
    }
}
