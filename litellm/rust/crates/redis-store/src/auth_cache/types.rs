use std::{collections::HashMap, time::Duration};

use jiff::Timestamp;
use litellm_python_compat::{
    Value,
    datetime::{IsoDateTime, UtcDateTime, fromisoformat},
};

use crate::{
    DecodeError, FromValue, IntoValue, Number, codec::python_value, dual_cache::MemorySettings,
};

/// `UserApiKeyCache`'s memory configuration: each partition's size and `default_in_memory_ttl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    pub memory_entries: usize,
    pub memory_ttl: Option<Duration>,
}

impl Default for Settings {
    /// `InMemoryCache`'s 200 entries.
    fn default() -> Self {
        Self {
            memory_entries: 200,
            memory_ttl: None,
        }
    }
}

impl Settings {
    pub(super) fn memory(self) -> MemorySettings {
        MemorySettings {
            entries: Some(self.memory_entries),
            ttl: self.memory_ttl,
        }
    }
}

pub(super) fn decode(raw: Option<Vec<u8>>) -> Result<Option<Value>, DecodeError> {
    raw.as_deref().map(python_value).transpose()
}

/// A JSON object the model types as a `dict`, kept as the plain data it was read as. It may hold
/// credentials, so `Debug` shows none of it.
#[derive(Clone, Default, PartialEq, veil::Redact)]
pub struct Bag(#[redact(with = "[REDACTED]")] Vec<(Value, Value)>);

impl Bag {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn entries(&self) -> &[(Value, Value)] {
        &self.0
    }
}

/// A `dict[str, list[str]]`, in the order it was written.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lists(pub Vec<(String, Vec<String>)>);

/// The two roles `Member` takes.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr, strum::VariantArray,
)]
pub enum MemberRole {
    #[strum(serialize = "admin")]
    Admin,
    #[strum(serialize = "user")]
    User,
}

/// One field of a cached record, read from and written as the plain data pydantic's JSON mode
/// dumps for it.
trait Field: Sized {
    fn read(value: &Value) -> Option<Self>;
    fn write(self) -> Value;
}

impl Field for Value {
    fn read(value: &Value) -> Option<Self> {
        Some(value.clone())
    }

    fn write(self) -> Value {
        self
    }
}

impl Field for Bag {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::Dict(entries) => Some(Self(entries.clone())),
            _ => None,
        }
    }

    fn write(self) -> Value {
        Value::Dict(self.0)
    }
}

impl Field for Lists {
    fn read(value: &Value) -> Option<Self> {
        let Value::Dict(entries) = value else {
            return None;
        };
        entries
            .iter()
            .map(|(name, list)| Some((String::read(name)?, Vec::read(list)?)))
            .collect::<Option<_>>()
            .map(Self)
    }

    fn write(self) -> Value {
        Value::Dict(
            self.0
                .into_iter()
                .map(|(name, list)| (Value::Str(name), list.write()))
                .collect(),
        )
    }
}

impl FromValue for Bag {
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        Self::read(value).ok_or(DecodeError::Shape("a dict"))
    }
}

impl FromValue for Lists {
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        Self::read(value).ok_or(DecodeError::Shape("a dict of string lists"))
    }
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

impl Field for String {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::Str(text) => Some(text.clone()),
            _ => None,
        }
    }

    fn write(self) -> Value {
        Value::Str(self)
    }
}

impl Field for bool {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::Bool(flag) => Some(*flag),
            _ => None,
        }
    }

    fn write(self) -> Value {
        Value::Bool(self)
    }
}

impl Field for i64 {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::Int(number) => i64::try_from(number).ok(),
            _ => None,
        }
    }

    fn write(self) -> Value {
        Value::from(self)
    }
}

/// A `float` field also takes the JSON integer pydantic accepts for it.
impl Field for f64 {
    fn read(value: &Value) -> Option<Self> {
        match Number::from_value(value)? {
            Number::Float(number) => Some(number),
            Number::Int(number) => Some(number as f64),
        }
    }

    fn write(self) -> Value {
        Value::Float(self)
    }
}

/// A `datetime` field. Pydantic writes an aware UTC one as `2026-11-19T10:00:00.123000Z`, with
/// no fraction when it is zero. One written without an offset is read as UTC.
impl Field for Timestamp {
    fn read(value: &Value) -> Option<Self> {
        let Value::Str(text) = value else {
            return None;
        };
        let (IsoDateTime::Aware(microseconds) | IsoDateTime::Naive(microseconds)) =
            fromisoformat(text)?;
        Timestamp::from_microsecond(microseconds).ok()
    }

    fn write(self) -> Value {
        let microseconds = self.as_microsecond();
        let at = UtcDateTime::from_timestamp(
            microseconds.div_euclid(1_000_000),
            microseconds.rem_euclid(1_000_000) as u32,
        );
        let fraction = match at.microsecond {
            0 => String::new(),
            microsecond => format!(".{microsecond:06}"),
        };
        Value::Str(format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{fraction}Z",
            at.year, at.month, at.day, at.hour, at.minute, at.second
        ))
    }
}

impl<T: Field> Field for Option<T> {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::None => Some(None),
            value => T::read(value).map(Some),
        }
    }

    fn write(self) -> Value {
        self.map_or(Value::None, T::write)
    }
}

impl<T: Field> Field for Vec<T> {
    fn read(value: &Value) -> Option<Self> {
        match value {
            Value::List(items) => items.iter().map(T::read).collect(),
            _ => None,
        }
    }

    fn write(self) -> Value {
        Value::List(self.into_iter().map(T::write).collect())
    }
}

/// The string-keyed entries of a dumped model, by field name.
fn fields(entries: &[(Value, Value)]) -> HashMap<&str, &Value> {
    entries
        .iter()
        .filter_map(|(key, value)| match key {
            Value::Str(key) => Some((key.as_str(), value)),
            _ => None,
        })
        .collect()
}

/// The default of a field the model requires, which an entry without it does not decode to.
macro_rules! required {
    ($field:literal) => {
        return Err(DecodeError::Shape($field))
    };
}

/// A record the auth cache holds as a pydantic model's JSON dump. The fields are that model's,
/// in its order, which is the order they are written in, and each takes only the shapes the
/// model takes, so nothing the model would refuse decodes or is written. A field the entry lacks
/// takes the model's default, and one this build does not name is kept in `unread` and written
/// after the named ones, so an entry a newer pod wrote loses nothing when it is read. A field
/// held as plain data may hold credentials, so `Debug` redacts it.
macro_rules! record {
    (
        $(#[$meta:meta])*
        pub struct $name:ident {
            $($(#[$field_meta:meta])* $field:ident: $kind:ty $(= $default:expr)?,)+
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, PartialEq, veil::Redact)]
        pub struct $name {
            $($(#[$field_meta])* pub $field: $kind,)+
            #[redact(with = "[REDACTED]")]
            pub unread: Vec<(String, Value)>,
        }

        impl FromValue for $name {
            fn from_value(value: &Value) -> Result<Self, DecodeError> {
                let Value::Dict(entries) = value else {
                    return Err(DecodeError::Shape(stringify!($name)));
                };
                let mut unnamed = fields(entries);
                Ok(Self {
                    $($field: match unnamed.remove(stringify!($field)) {
                        None => record!(@default $($default)?),
                        Some(value) => <$kind as Field>::read(value)
                            .ok_or(DecodeError::Shape(stringify!($field)))?,
                    },)+
                    unread: entries
                        .iter()
                        .filter_map(|(key, value)| match key {
                            Value::Str(key) if unnamed.contains_key(key.as_str()) => {
                                Some((key.clone(), value.clone()))
                            }
                            _ => None,
                        })
                        .collect(),
                })
            }
        }

        impl IntoValue for $name {
            fn into_value(self) -> Value {
                Value::Dict(
                    [$((stringify!($field).to_owned(), Field::write(self.$field)),)+]
                        .into_iter()
                        .chain(self.unread)
                        .map(|(name, value)| (Value::Str(name), value))
                        .collect(),
                )
            }
        }

        impl Field for $name {
            fn read(value: &Value) -> Option<Self> {
                Self::from_value(value).ok()
            }

            fn write(self) -> Value {
                self.into_value()
            }
        }
    };
    (@default) => {
        Default::default()
    };
    (@default $default:expr) => {
        $default
    };
}

record! {
    /// A team member as `Member` dumps it inside a cached key.
    pub struct CachedMember {
        user_id: Option<String>,
        user_email: Option<String>,
        role: MemberRole = required!("role"),
    }
}

record! {
    /// `LiteLLM_ObjectPermissionTable.model_dump(mode="json")`, the value under
    /// [`AuthObject::ObjectPermission`](super::AuthObject::ObjectPermission) and nested in a
    /// cached key.
    pub struct CachedObjectPermission {
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
    }
}

record! {
    /// `UserAPIKeyAuth.model_dump(mode="json")`, the value under
    /// [`AuthObject::Key`](super::AuthObject::Key). A key read from the database is cached in
    /// this shape too, so it is what a key read returns from either tier. `expires` stays the
    /// text it was written as, since the model types it `str | datetime` and Python before 3.11
    /// cannot read a `Z` suffix back.
    pub struct CachedKey {
        key_id: Option<String>,
        token: Option<String>,
        key_name: Option<String>,
        key_alias: Option<String>,
        spend: f64,
        total_spend: f64,
        max_budget: Option<f64>,
        expires: Option<String>,
        models: Vec<String>,
        aliases: Bag,
        config: Bag,
        user_id: Option<String>,
        team_id: Option<String>,
        agent_id: Option<String>,
        project_id: Option<String>,
        max_parallel_requests: Option<i64>,
        metadata: Bag,
        tpm_limit: Option<i64>,
        rpm_limit: Option<i64>,
        tpd_limit: Option<i64>,
        budget_duration: Option<String>,
        budget_reset_at: Option<Timestamp>,
        allowed_cache_controls: Option<Vec<String>> = Some(Vec::new()),
        allowed_routes: Option<Vec<String>> = Some(Vec::new()),
        key_type: Option<String>,
        permissions: Bag,
        model_spend: Bag,
        model_max_budget: Bag,
        budget_fallbacks: Lists,
        soft_budget_cooldown: bool,
        blocked: Option<bool>,
        litellm_budget_table: Option<Bag>,
        budget_id: Option<String>,
        org_id: Option<String>,
        created_at: Option<Timestamp>,
        created_by: Option<String>,
        updated_at: Option<Timestamp>,
        updated_by: Option<String>,
        settings_updated_at: Option<Timestamp>,
        last_active: Option<Timestamp>,
        object_permission_id: Option<String>,
        object_permission: Option<CachedObjectPermission>,
        access_group_ids: Option<Vec<String>>,
        rotation_count: Option<i64> = Some(0),
        auto_rotate: Option<bool> = Some(false),
        rotation_interval: Option<String>,
        last_rotation_at: Option<Timestamp>,
        key_rotation_at: Option<Timestamp>,
        router_settings: Option<Bag>,
        budget_limits: Option<Vec<Bag>>,
        team_spend: Option<f64>,
        team_alias: Option<String>,
        team_tpm_limit: Option<i64>,
        team_rpm_limit: Option<i64>,
        team_tpd_limit: Option<i64>,
        team_max_budget: Option<f64>,
        team_soft_budget: Option<f64>,
        team_model_max_budget: Option<Bag>,
        team_models: Vec<String>,
        team_blocked: bool,
        soft_budget: Option<f64>,
        team_model_aliases: Option<Bag>,
        team_member: Option<CachedMember>,
        team_metadata: Option<Bag>,
        team_object_permission_id: Option<String>,
        team_member_spend: Option<f64>,
        team_member_tpm_limit: Option<i64>,
        team_member_rpm_limit: Option<i64>,
        end_user_id: Option<String>,
        end_user_tpm_limit: Option<i64>,
        end_user_rpm_limit: Option<i64>,
        end_user_tpd_limit: Option<i64>,
        end_user_max_budget: Option<f64>,
        end_user_model_max_budget: Option<Bag>,
        end_user_model_tpm_limit: Option<Bag>,
        end_user_model_rpm_limit: Option<Bag>,
        #[redact(with = "[REDACTED]")]
        customer_hierarchy: Option<Value>,
        organization_alias: Option<String>,
        organization_max_budget: Option<f64>,
        organization_tpm_limit: Option<i64>,
        organization_rpm_limit: Option<i64>,
        organization_metadata: Option<Bag>,
        project_alias: Option<String>,
        project_metadata: Option<Bag>,
        last_refreshed_at: Option<f64>,
        api_key: Option<String>,
        user_role: Option<String>,
        allowed_model_region: Option<String>,
        #[redact(with = "[REDACTED]")]
        parent_otel_span: Option<Value>,
        rpm_limit_per_model: Option<Bag>,
        tpm_limit_per_model: Option<Bag>,
        user_tpm_limit: Option<i64>,
        user_rpm_limit: Option<i64>,
        user_email: Option<String>,
        user_spend: Option<f64>,
        user_max_budget: Option<f64>,
        user_model_max_budget: Option<Bag>,
        request_route: Option<String>,
        is_session_token: bool,
        #[redact(with = "[REDACTED]")]
        user: Option<Value>,
        #[redact(with = "[REDACTED]")]
        created_by_user: Option<Value>,
        end_user_object_permission: Option<CachedObjectPermission>,
        team_object_permission: Option<CachedObjectPermission>,
        jwt_claims: Option<Bag>,
    }
}

#[cfg(test)]
mod tests {
    use jiff::Timestamp;
    use litellm_python_compat::{Value, json};
    use rstest::rstest;

    use super::{CachedKey, CachedMember, CachedObjectPermission};
    use crate::{DecodeError, FromValue, IntoValue};

    fn key(entry: &str) -> Result<CachedKey, DecodeError> {
        CachedKey::from_value(&json::loads(entry).unwrap())
    }

    #[rstest]
    fn a_field_the_entry_lacks_takes_the_model_default() {
        let key = key(r#"{"token": "hashed"}"#).unwrap();

        assert_eq!(
            (
                key.allowed_cache_controls,
                key.allowed_routes,
                key.rotation_count,
                key.auto_rotate,
                key.aliases.is_empty(),
                key.access_group_ids,
                key.router_settings,
                key.spend,
                key.team_blocked,
            ),
            (
                Some(Vec::new()),
                Some(Vec::new()),
                Some(0),
                Some(false),
                true,
                None,
                None,
                0.0,
                false,
            )
        );
    }

    #[rstest]
    fn a_field_this_build_does_not_name_is_kept_and_written_after_the_named_ones() {
        let key =
            key(r#"{"future_limit": {"rpm": 7}, "token": "hashed", "also_new": null}"#).unwrap();

        let unread = key.unread.clone();
        let written = json::dumps(&key.into_value()).unwrap();

        assert_eq!(
            (
                unread,
                written.starts_with(r#"{"key_id": null, "token": "hashed", "#),
                written.ends_with(
                    r#""jwt_claims": null, "future_limit": {"rpm": 7}, "also_new": null}"#
                ),
            ),
            (
                vec![
                    (
                        "future_limit".to_owned(),
                        json::loads(r#"{"rpm": 7}"#).unwrap()
                    ),
                    ("also_new".to_owned(), Value::None),
                ],
                true,
                true,
            )
        );
    }

    #[rstest]
    #[case::utc_as_pydantic_writes_it(
        r#""2026-11-19T10:00:00.123000Z""#,
        "2026-11-19T10:00:00.123Z"
    )]
    #[case::whole_seconds(r#""2026-11-19T10:00:00Z""#, "2026-11-19T10:00:00Z")]
    #[case::another_offset(r#""2026-11-19T19:00:00+09:00""#, "2026-11-19T10:00:00Z")]
    #[case::no_offset_is_utc(r#""2026-11-19T10:00:00.5""#, "2026-11-19T10:00:00.5Z")]
    fn a_time_python_wrote_is_its_instant(#[case] written: &str, #[case] instant: &str) {
        let key = key(&format!(r#"{{"created_at": {written}}}"#)).unwrap();

        assert_eq!(key.created_at, Some(instant.parse::<Timestamp>().unwrap()));
    }

    #[rstest]
    #[case::a_fraction("2026-11-19T10:00:00.123Z", r#""2026-11-19T10:00:00.123000Z""#)]
    #[case::whole_seconds("2026-11-19T10:00:00Z", r#""2026-11-19T10:00:00Z""#)]
    #[case::before_the_epoch("1969-12-31T23:59:59.5Z", r#""1969-12-31T23:59:59.500000Z""#)]
    fn a_time_is_written_as_pydantic_writes_an_aware_utc_datetime(
        #[case] instant: &str,
        #[case] written: &str,
    ) {
        let key = CachedKey {
            created_at: Some(instant.parse().unwrap()),
            ..key("{}").unwrap()
        };

        let entry = json::dumps(&key.into_value()).unwrap();

        assert!(
            entry.contains(&format!(r#""created_at": {written},"#)),
            "{entry}"
        );
    }

    #[rstest]
    #[case::a_number_that_is_text(r#"{"spend": "free"}"#, "spend")]
    #[case::a_required_number_that_is_null(r#"{"spend": null}"#, "spend")]
    #[case::a_limit_that_is_a_fraction(r#"{"rpm_limit": 1.5}"#, "rpm_limit")]
    #[case::a_time_that_is_no_time(r#"{"created_at": "tomorrow"}"#, "created_at")]
    #[case::a_model_list_holding_a_number(r#"{"models": ["a", 1]}"#, "models")]
    #[case::a_member_without_a_role(r#"{"team_member": {"user_id": "u"}}"#, "team_member")]
    #[case::a_member_with_another_role(
        r#"{"team_member": {"user_id": "u", "role": "owner"}}"#,
        "team_member"
    )]
    #[case::metadata_that_is_a_list(r#"{"metadata": []}"#, "metadata")]
    #[case::metadata_that_is_null(r#"{"metadata": null}"#, "metadata")]
    #[case::fallbacks_that_are_not_lists(r#"{"budget_fallbacks": {"m": "n"}}"#, "budget_fallbacks")]
    #[case::budget_limits_that_are_no_list(r#"{"budget_limits": {"m": 1}}"#, "budget_limits")]
    #[case::router_settings_that_are_text(r#"{"router_settings": "{}"}"#, "router_settings")]
    #[case::a_permission_whose_tool_map_is_text(
        r#"{"object_permission": {"object_permission_id": "p", "mcp_tool_permissions": "{\"s\": [\"t\"]}"}}"#,
        "object_permission"
    )]
    #[case::a_permission_without_its_id(
        r#"{"object_permission": {"models": []}}"#,
        "object_permission"
    )]
    fn an_entry_the_model_would_refuse_does_not_decode(#[case] entry: &str, #[case] field: &str) {
        assert!(
            matches!(key(entry), Err(DecodeError::Shape(refused)) if refused == field),
            "{:?}",
            key(entry).err()
        );
    }

    #[rstest]
    fn a_float_field_takes_the_integer_python_also_takes() {
        assert_eq!(
            key(r#"{"spend": 3, "max_budget": 10}"#)
                .map(|key| (key.spend, key.max_budget))
                .unwrap(),
            (3.0, Some(10.0))
        );
    }

    #[rstest]
    fn debug_shows_no_settings_bag() {
        let key = key(concat!(
            r#"{"token": "hashed", "metadata": {"api_key": "credential-value"}, "#,
            r#""team_member": {"user_id": "u", "role": "user", "note": "credential-value"}, "#,
            r#""router_settings": {"redis_password": "credential-value"}, "budget_limits": [{"k": "credential-value"}], "#,
            r#""later_field": "credential-value"}"#,
        ))
        .unwrap();

        let debug = format!("{key:?}");

        assert_eq!(
            (
                debug.contains("credential-value"),
                debug.contains("[REDACTED]"),
                debug.contains("hashed"),
            ),
            (false, true, true)
        );
    }

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
