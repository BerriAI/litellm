//! The end user a request names, read through the store and turned into the core's record. The
//! native app charges and limits no end user, so a row that holds anything the native app does not
//! apply yet is declined, naming the field.

use litellm_gateway_auth_core::{Error, record::EndUserRecord};
use litellm_redis_store::auth_cache::EndUserEntry;

use super::{
    AuthStore,
    carried::{Carried, carried, newer},
    store::EndUserRead,
};

/// The end user `end_user_id` as the native app serves it: one with no row, or a row that only
/// describes it. With no database no end user has a row, so the request is served under the id
/// alone. A row the database could not give sends the request to Python.
pub(crate) async fn end_user(
    store: Option<&AuthStore>,
    end_user_id: &str,
) -> Result<EndUserRecord, Error> {
    let unrestricted = || EndUserRecord {
        end_user_id: Some(end_user_id.to_owned()),
        ..EndUserRecord::default()
    };
    let Some(store) = store else {
        return Ok(unrestricted());
    };
    match store.end_user(end_user_id).await {
        EndUserRead::NoRow => Ok(unrestricted()),
        EndUserRead::Row(end_user) => record(&end_user).map_err(|field| {
            tracing::debug!(
                field,
                "the end user's row holds what only the proxy applies"
            );
            Error::HostDeclined
        }),
        EndUserRead::Unread => Err(Error::HostFailed),
    }
}

/// The entry is taken apart by name, so a field added to it does not compile here until it is
/// read for nothing or declined. The alias and the spend describe the end user. Every other field
/// that holds something is a budget, a model list, a region, a block or a grant the native app
/// does not apply.
fn record(end_user: &EndUserEntry) -> Result<EndUserRecord, &'static str> {
    let EndUserEntry {
        user_id,
        blocked,
        alias: _,
        spend: _,
        allowed_model_region,
        default_model,
        models,
        budget_id,
        litellm_budget_table,
        object_permission_id,
        object_permission,
        unread,
    } = end_user;
    let unread = &newer(unread);
    let held = carried![
        blocked,
        allowed_model_region,
        default_model,
        models,
        budget_id,
        litellm_budget_table,
        object_permission_id,
        object_permission,
        unread,
    ];
    match held {
        Some(field) => Err(field),
        None => Ok(EndUserRecord {
            end_user_id: Some(user_id.clone()),
            found: true,
            ..EndUserRecord::default()
        }),
    }
}

#[cfg(test)]
mod tests {
    use litellm_redis_store::FromValue;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::host::auth::keys;

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn an_end_user_the_database_cannot_give_is_left_to_python() {
        let store = AuthStore::new(keys::memory_cache(), keys::unreachable());

        assert_eq!(end_user(Some(&store), "eu-1").await, Err(Error::HostFailed));
    }

    fn stored(columns: Value) -> EndUserEntry {
        let mut row = json!({"user_id": "eu-1", "blocked": false, "spend": 0.0, "models": []});
        row.as_object_mut()
            .unwrap()
            .extend(columns.as_object().unwrap().clone());
        EndUserEntry::from_value(&litellm_python_compat::json::from_json(row)).unwrap()
    }

    #[rstest]
    #[case::nothing_else(json!({}))]
    #[case::what_it_spent_and_is_called(json!({"spend": 12.5, "alias": "acme"}))]
    fn a_row_that_only_describes_the_end_user_is_served(#[case] columns: Value) {
        assert_eq!(
            record(&stored(columns)),
            Ok(EndUserRecord {
                end_user_id: Some("eu-1".to_owned()),
                found: true,
                ..EndUserRecord::default()
            })
        );
    }

    #[rstest]
    #[case::a_block(json!({"blocked": true}), "blocked")]
    #[case::a_region(json!({"allowed_model_region": "eu"}), "allowed_model_region")]
    #[case::a_default_model(json!({"default_model": "claude"}), "default_model")]
    #[case::a_model_list(json!({"models": ["claude"]}), "models")]
    #[case::a_budget_it_names(json!({"budget_id": "b-1"}), "budget_id")]
    #[case::a_budget_it_holds(
        json!({"litellm_budget_table": {"max_budget": 1.0}}),
        "litellm_budget_table"
    )]
    #[case::an_object_permission_it_names(
        json!({"object_permission_id": "p-1"}),
        "object_permission_id"
    )]
    #[case::an_object_permission_it_holds(
        json!({"object_permission": {"object_permission_id": "p-1"}}),
        "object_permission"
    )]
    #[case::a_field_this_build_does_not_name(json!({"future_limit": 1}), "unread")]
    fn a_row_that_holds_what_the_native_app_does_not_apply_is_declined_by_its_field(
        #[case] columns: Value,
        #[case] field: &str,
    ) {
        assert_eq!(record(&stored(columns)), Err(field));
    }
}
