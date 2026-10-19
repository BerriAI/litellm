#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::Display,
    strum::EnumString,
    strum::AsRefStr,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(from = "String", into = "String"))]
pub enum MessageRole {
    #[strum(serialize = "user")]
    User,
    #[strum(serialize = "assistant")]
    Assistant,
    #[strum(serialize = "system")]
    System,
    #[strum(default, transparent)]
    Other(String),
}

impl From<String> for MessageRole {
    fn from(value: String) -> Self {
        value.parse().unwrap_or_else(|never| match never {})
    }
}

impl From<MessageRole> for String {
    fn from(value: MessageRole) -> Self {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::MessageRole;
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::user("user", MessageRole::User)]
    #[case::assistant("assistant", MessageRole::Assistant)]
    #[case::system("system", MessageRole::System)]
    #[case::unknown("future_role", MessageRole::Other("future_role".into()))]
    #[case::case_sensitive("User", MessageRole::Other("User".into()))]
    #[case::empty("", MessageRole::Other(String::new()))]
    fn roles_preserve_wire_values(#[case] wire: &str, #[case] expected: MessageRole) {
        let role: MessageRole = serde_json::from_value(json!(wire)).unwrap();

        assert_eq!(role, expected);
        assert_eq!(wire.parse::<MessageRole>().unwrap(), role);
        assert_eq!(role.as_ref(), wire);
        assert_eq!(serde_json::to_value(role).unwrap(), json!(wire));
    }

    #[rstest]
    fn malformed_roles_are_rejected(
        #[values(json!(7), json!(true), json!([]), json!({}), Value::Null)] wire: Value,
    ) {
        assert!(serde_json::from_value::<MessageRole>(wire).is_err());
    }

    #[cfg(feature = "schema")]
    #[rstest]
    fn role_schema_accepts_unknown_strings() {
        let schema = schemars::schema_for!(MessageRole).to_value();

        assert_eq!(schema.get("type"), Some(&json!("string")));
        assert!(schema.get("enum").is_none());
    }
}
