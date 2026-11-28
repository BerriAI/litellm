use litellm_python_compat::{Value, datetime::IsoDateTime};

use super::{
    bag::Bag,
    budget::BudgetEntry,
    record::{record, required},
};
use crate::{DecodeError, FromValue, IntoValue};

/// `TAG_REGISTRY_OVERFLOW_SENTINEL`.
const UNUSABLE: &str = "__tag_registry_overflow__";

record! {
    /// `LiteLLM_TagTable.model_dump(mode="json")`, the value under
    /// [`AuthObject::Tag`](crate::auth_cache::AuthObject::Tag).
    pub struct TagEntry {
        tag_name: String = required!("tag_name"),
        description: Option<String>,
        models: Vec<String> => default_on_null,
        model_info: Option<Bag>,
        spend: f64 => default_on_null,
        budget_id: Option<String>,
        litellm_budget_table: Option<BudgetEntry>,
        created_at: Option<IsoDateTime>,
        created_by: Option<String>,
        updated_at: Option<IsoDateTime>,
    }
}

/// The value under [`AuthObject::TagRegistry`](crate::auth_cache::AuthObject::TagRegistry), as
/// `_load_tag_registry` caches it: the name of every tag that has a row, or the word Python
/// caches when it cannot hold them, with more rows than `TAG_REGISTRY_MAX_SIZE` or a database
/// that did not list them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TagNames {
    Listed(Vec<String>),
    Unusable,
}

impl FromValue for TagNames {
    /// `_cached_registry`: an entry that is no string is left out, and any other value is no
    /// registry.
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        match value {
            Value::Str(word) if word == UNUSABLE => Ok(Self::Unusable),
            Value::List(names) | Value::Tuple(names) => Ok(Self::Listed(
                names
                    .iter()
                    .filter_map(|name| match name {
                        Value::Str(name) => Some(name.clone()),
                        _ => None,
                    })
                    .collect(),
            )),
            _ => Err(DecodeError::Shape("TagNames")),
        }
    }
}

impl IntoValue for TagNames {
    fn into_value(self) -> Value {
        match self {
            Self::Listed(names) => Value::List(names.into_iter().map(Value::Str).collect()),
            Self::Unusable => Value::Str(UNUSABLE.to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use rstest::rstest;

    use super::TagNames;
    use crate::FromValue;

    #[rstest]
    #[case::an_entry_that_is_no_string_is_left_out(
        r#"[1, "plain", null]"#,
        Some(TagNames::Listed(vec!["plain".to_owned()]))
    )]
    #[case::another_word(r#""__end_user_restricted_registry_overflow__""#, None)]
    #[case::a_tag_entry(r#"{"tag_name": "plain"}"#, None)]
    fn tag_names_are_read_as_python_reads_its_registry(
        #[case] cached: &str,
        #[case] names: Option<TagNames>,
    ) {
        assert_eq!(
            TagNames::from_value(&json::loads(cached).unwrap()).ok(),
            names
        );
    }
}
