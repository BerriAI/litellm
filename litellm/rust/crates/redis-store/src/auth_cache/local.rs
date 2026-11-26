//! Local auth-cache values and the index used for compact team reads.

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use litellm_python_compat::Value;
use parking_lot::Mutex;

use super::MemberEntry;
use crate::{DecodeError, FromValue, IntoValue, Number};

/// The wire value and the local-only state held for exactly as long as the cache entry.
#[derive(Clone)]
pub(super) struct LocalValue(Arc<Entry>);

/// A roster reference bound to the cache entry this request read.
pub struct TeamRoster(LocalValue);

impl TeamRoster {
    pub fn member(&self, user_id: &str) -> Option<Value> {
        let value = self.0.0.value.lock();
        let Value::Dict(row) = &*value else {
            return None;
        };
        let index = self.0.0.members.get()?.as_ref()?;
        member(row, index, user_id).cloned()
    }
}

struct Entry {
    value: Mutex<Value>,
    members: OnceLock<Option<HashMap<String, usize>>>,
}

impl From<Value> for LocalValue {
    fn from(value: Value) -> Self {
        Self(Arc::new(Entry {
            value: Mutex::new(value),
            members: OnceLock::new(),
        }))
    }
}

impl FromValue for LocalValue {
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        Ok(value.clone().into())
    }
}

impl IntoValue for LocalValue {
    fn into_value(self) -> Value {
        self.0.value.lock().clone()
    }
}

impl LocalValue {
    pub(super) fn team_roster(&self) -> Option<TeamRoster> {
        let value = self.0.value.lock();
        let Value::Dict(row) = &*value else {
            return None;
        };
        self.0.members.get_or_init(|| member_index(row)).as_ref()?;
        Some(TeamRoster(self.clone()))
    }

    pub(super) fn team_fields(self, fields: Option<&[String]>, user_id: Option<&str>) -> Value {
        let value = self.0.value.lock();
        let Value::Dict(row) = &*value else {
            return value.clone();
        };
        Value::Dict(match fields {
            Some(fields) => row
                .iter()
                .filter(|(key, _)| fields.iter().any(|field| named(key, field)))
                .cloned()
                .collect::<Vec<_>>(),
            None => {
                let Some(index) = self.0.members.get_or_init(|| member_index(row)) else {
                    return value.clone();
                };
                let mut result = row
                    .iter()
                    .filter(|(key, _)| {
                        !["members_with_roles", "members", "admins"]
                            .iter()
                            .any(|n| named(key, n))
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                if let Some(member) = user_id.and_then(|id| member(row, index, id)) {
                    result.push((
                        Value::Str("members_with_roles".into()),
                        Value::List(vec![member.clone()]),
                    ));
                }
                result
            }
        })
    }

    pub(super) fn add_team_spend(&self, delta: f64) -> bool {
        let mut value = self.0.value.lock();
        let Value::Dict(fields) = &mut *value else {
            return false;
        };
        let Some((_, spend)) = fields.iter_mut().find(|(key, _)| named(key, "spend")) else {
            fields.push((Value::Str("spend".into()), Value::Float(delta)));
            return true;
        };
        let initial = match &*spend {
            Value::None => 0.0,
            Value::Bool(value) => f64::from(u8::from(*value)),
            Value::Str(value) => match value.trim().parse::<f64>() {
                Ok(value) => value,
                Err(_) => return false,
            },
            value => match Number::from_value(value) {
                Some(value) => value.as_f64(),
                None => return false,
            },
        };
        *spend = Value::Float(initial + delta);
        true
    }
}

fn member<'a>(
    row: &'a [(Value, Value)],
    index: &HashMap<String, usize>,
    id: &str,
) -> Option<&'a Value> {
    let position = *index.get(id)?;
    match field(row, "members_with_roles")? {
        Value::List(members) | Value::Tuple(members) => members.get(position),
        _ => None,
    }
}

fn named(key: &Value, name: &str) -> bool {
    matches!(key, Value::Str(key) if key == name)
}

fn field<'a>(row: &'a [(Value, Value)], name: &str) -> Option<&'a Value> {
    row.iter()
        .find(|(key, _)| named(key, name))
        .map(|(_, value)| value)
}

fn member_index(row: &[(Value, Value)]) -> Option<HashMap<String, usize>> {
    for name in ["members", "admins"] {
        match field(row, name) {
            None => (),
            Some(Value::List(ids) | Value::Tuple(ids))
                if ids.iter().all(|id| matches!(id, Value::Str(_))) => {}
            _ => return None,
        }
    }
    let members = match field(row, "members_with_roles") {
        None => return Some(HashMap::new()),
        Some(Value::Dict(members)) if members.is_empty() => return Some(HashMap::new()),
        Some(Value::List(members) | Value::Tuple(members)) => members,
        _ => return None,
    };
    let mut index = HashMap::new();
    for (position, value) in members.iter().enumerate() {
        let member = MemberEntry::from_value(value).ok()?;
        if let Some(user_id) = member.user_id {
            index.entry(user_id).or_insert(position);
        }
    }
    Some(index)
}
