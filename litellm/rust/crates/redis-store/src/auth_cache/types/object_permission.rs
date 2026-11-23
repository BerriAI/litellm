use litellm_python_compat::Value;

use super::{
    bag::Lists,
    record::{Field, record, required},
};

/// Stored `mcp_auth_profiles` state: `Absent` (unread) vs `Empty` (read, no grants).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ProfileGrants {
    #[default]
    Absent,
    Empty,
    Named(Vec<(String, String)>),
    Unreadable,
}

impl From<&Value> for ProfileGrants {
    fn from(value: &Value) -> Self {
        match value {
            Value::None => Self::Empty,
            Value::Dict(entries) if entries.is_empty() => Self::Empty,
            Value::Dict(entries) => entries
                .iter()
                .map(|(server, profile)| Some((String::read(server)?, String::read(profile)?)))
                .collect::<Option<_>>()
                .map_or(Self::Unreadable, Self::Named),
            _ => Self::Unreadable,
        }
    }
}

impl Field for ProfileGrants {
    fn read(value: &Value) -> Option<Self> {
        Some(value.into())
    }

    fn write(self) -> Value {
        match self {
            Self::Absent | Self::Empty => Value::None,
            Self::Named(entries) => Value::Dict(
                entries
                    .into_iter()
                    .map(|(k, v)| (k.write(), v.write()))
                    .collect(),
            ),
            Self::Unreadable => Value::Str("unreadable".into()),
        }
    }

    fn dump(self) -> Option<Value> {
        (!matches!(self, Self::Absent)).then(|| self.write())
    }
}

record! {
    /// `LiteLLM_ObjectPermissionTable.model_dump(mode="json")`, the value under
    /// [`AuthObject::ObjectPermission`](crate::auth_cache::AuthObject::ObjectPermission) and
    /// nested in a key's, a team's, a project's, an end user's, an organization's or a user's entry.
    pub struct ObjectPermissionEntry {
        object_permission_id: String = required!("object_permission_id"),
        mcp_servers: Option<Vec<String>> = Some(Vec::new()),
        mcp_access_groups: Option<Vec<String>> = Some(Vec::new()),
        mcp_tool_permissions: Option<Lists>,
        vector_stores: Option<Vec<String>> = Some(Vec::new()),
        agents: Option<Vec<String>> = Some(Vec::new()),
        agent_access_groups: Option<Vec<String>> = Some(Vec::new()),
        models: Option<Vec<String>> = Some(Vec::new()),
        mcp_toolsets: Option<Vec<String>>,
        blocked_tools: Option<Vec<String>> = Some(Vec::new()),
        search_tools: Option<Vec<String>> = Some(Vec::new()),
        mcp_tool_search_enabled: Option<bool>,
        skills: Option<Vec<String>>,
        apps: Option<Vec<String>>,
        mcp_auth_profiles: ProfileGrants,
    }
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::{Value, json};
    use rstest::rstest;

    use super::{ObjectPermissionEntry, ProfileGrants};
    use crate::{FromValue, IntoValue, auth_cache::KeyEntry};

    #[rstest]
    #[case::bad_value(r#"{"server": 1}"#)]
    #[case::list(r#"["server"]"#)]
    #[case::cached_refusal(r#""unreadable""#)]
    fn unreadable_grants_round_trip_with_unrelated_permissions(#[case] stored: &str) {
        let input = format!(
            r#"{{"object_permission": {{"object_permission_id":"p","vector_stores":["v"],"mcp_auth_profiles":{stored}}}}}"#
        );
        let original = KeyEntry::from_value(&json::loads(&input).unwrap()).unwrap();
        let cached = KeyEntry::from_value(&original.into_value()).unwrap();
        let permission = cached.object_permission.unwrap();
        assert_eq!(permission.vector_stores, Some(vec!["v".into()]));
        assert_eq!(permission.mcp_auth_profiles, ProfileGrants::Unreadable);
    }

    #[rstest]
    #[case::never_read(None, ProfileGrants::Absent, None)]
    #[case::null(Some("null"), ProfileGrants::Empty, Some(Value::None))]
    #[case::empty(Some("{}"), ProfileGrants::Empty, Some(Value::None))]
    fn a_grant_column_read_empty_is_written_as_null_and_one_never_read_is_left_out(
        #[case] stored: Option<&str>,
        #[case] read: ProfileGrants,
        #[case] written: Option<Value>,
    ) {
        let grants = stored.map_or_else(String::new, |value| {
            format!(r#","mcp_auth_profiles":{value}"#)
        });
        let permission = ObjectPermissionEntry::from_value(
            &json::loads(&format!(r#"{{"object_permission_id":"p"{grants}}}"#)).unwrap(),
        )
        .unwrap();
        let state = permission.mcp_auth_profiles.clone();
        let rewritten = match permission.into_value() {
            Value::Dict(entries) => entries.into_iter().find_map(|(name, value)| {
                (name == Value::Str("mcp_auth_profiles".into())).then_some(value)
            }),
            _ => None,
        };

        assert_eq!((state, rewritten), (read, written));
    }
}
