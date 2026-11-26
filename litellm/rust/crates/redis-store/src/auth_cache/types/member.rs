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
    /// A team member as `Member` dumps it inside a cached key.
    pub struct CachedMember {
        user_id: Option<String>,
        user_email: Option<String>,
        role: MemberRole = required!("role"),
    }
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use rstest::rstest;

    use super::CachedMember;
    use crate::{FromValue, IntoValue, auth_cache::CachedObjectPermission};

    #[rstest]
    fn a_member_and_a_permission_keep_what_they_do_not_name() {
        let member = CachedMember::from_value(
            &json::loads(r#"{"user_id": "u", "role": "user", "joined": "2026"}"#).unwrap(),
        )
        .unwrap();
        let permission = CachedObjectPermission::from_value(
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
