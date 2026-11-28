use litellm_python_compat::Value;

use super::{
    budget::EndUserBudgetEntry,
    object_permission::ObjectPermissionEntry,
    record::{Literal, record, required},
};
use crate::{DecodeError, FromValue, IntoValue};

/// `END_USER_RESTRICTED_REGISTRY_OVERFLOW_SENTINEL`.
const OVERFLOW: &str = "__end_user_restricted_registry_overflow__";

/// The two regions `LiteLLM_EndUserTable.allowed_model_region` takes.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr, strum::VariantArray,
)]
#[strum(serialize_all = "lowercase")]
pub enum ModelRegion {
    Eu,
    Us,
}

impl Literal for ModelRegion {}

record! {
    /// `LiteLLM_EndUserTable.model_dump(mode="json")`, the value `get_end_user_object` caches
    /// under [`AuthObject::EndUser`](crate::auth_cache::AuthObject::EndUser).
    pub struct EndUserEntry {
        user_id: String = required!("user_id"),
        blocked: bool = required!("blocked"),
        alias: Option<String>,
        spend: f64,
        allowed_model_region: Option<ModelRegion>,
        default_model: Option<String>,
        models: Vec<String>,
        budget_id: Option<String>,
        litellm_budget_table: Option<EndUserBudgetEntry>,
        object_permission_id: Option<String>,
        object_permission: Option<ObjectPermissionEntry>,
    }
}

/// What `_load_end_user_restricted_registry` caches under
/// [`AuthObject::EndUserRestrictedRegistry`](crate::auth_cache::AuthObject::EndUserRestrictedRegistry):
/// the ids of the end users whose row restricts something, or `Overflow` when it could not list
/// them all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestrictedEndUsers {
    Ids(Vec<String>),
    Overflow,
}

/// `_cached_registry` keeps only the text items of a cached list.
impl FromValue for RestrictedEndUsers {
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        match value {
            Value::Str(word) if word == OVERFLOW => Ok(Self::Overflow),
            Value::List(ids) => Ok(Self::Ids(
                ids.iter()
                    .filter_map(|id| match id {
                        Value::Str(id) => Some(id.clone()),
                        _ => None,
                    })
                    .collect(),
            )),
            _ => Err(DecodeError::Shape("a list of end user ids or the overflow word")),
        }
    }
}

impl IntoValue for RestrictedEndUsers {
    fn into_value(self) -> Value {
        match self {
            Self::Ids(ids) => Value::List(ids.into_iter().map(Value::Str).collect()),
            Self::Overflow => Value::Str(OVERFLOW.to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::ids(RestrictedEndUsers::Ids(vec!["eu-1".to_owned(), "eu-2".to_owned()]))]
    #[case::none(RestrictedEndUsers::Ids(Vec::new()))]
    #[case::overflow(RestrictedEndUsers::Overflow)]
    fn the_restricted_ids_read_back_as_written(#[case] written: RestrictedEndUsers) {
        let value = written.clone().into_value();

        assert_eq!(RestrictedEndUsers::from_value(&value).ok(), Some(written));
    }

    #[rstest]
    fn a_cached_list_keeps_only_its_text_ids() {
        let cached = Value::List(vec![Value::Str("eu-1".to_owned()), Value::Bool(true)]);

        assert_eq!(
            RestrictedEndUsers::from_value(&cached).ok(),
            Some(RestrictedEndUsers::Ids(vec!["eu-1".to_owned()]))
        );
    }

    #[rstest]
    #[case::other_text(Value::Str("eu-1".to_owned()))]
    #[case::nothing(Value::None)]
    fn any_other_value_does_not_decode(#[case] cached: Value) {
        assert!(RestrictedEndUsers::from_value(&cached).is_err());
    }
}
