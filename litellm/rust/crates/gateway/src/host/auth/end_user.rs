//! The end user a request names, read through the store and turned into the core's record. The
//! native app charges and limits no end user. A column a plain value holds is copied onto the
//! record, and [`unserved`] names the first one nothing native reads yet, which `admit` sends the
//! request to Python over. A column that holds a structure, or a value whose type the record has no
//! counterpart for, is declined here, naming the field.

use litellm_gateway_auth_core::{
    Error,
    record::{Budget, EndUserRecord, MinuteLimits, ModelRegion},
};
use litellm_redis_store::auth_cache::{EndUserEntry, ModelRegion as StoredModelRegion};

use super::{
    AuthStore,
    affiliation::{BudgetRow, budget_row_or_spend, unserved_budget_extras},
    carried::{Carried, carried, newer},
    object_permission,
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
        EndUserRead::Row(end_user) => record(&end_user).or_else(super::declined),
        EndUserRead::Unread => Err(Error::HostFailed),
    }
}

/// The entry is taken apart by name, so a field added to it does not compile here until it is
/// copied onto the record, read for nothing or declined. The alias describes the end user. The
/// block, the default model, the model list and the plain values of its budget row, with the spend
/// against it, are copied onto the record.
fn record(end_user: &EndUserEntry) -> Result<EndUserRecord, &'static str> {
    let EndUserEntry {
        user_id,
        blocked,
        alias: _,
        spend,
        allowed_model_region,
        default_model,
        models,
        budget_id: _,
        litellm_budget_table,
        object_permission_id,
        object_permission,
        unread,
    } = end_user;
    let unread = &newer(unread);
    let held = carried![unread];
    if let Some(field) = held {
        return Err(field);
    }
    let BudgetRow {
        limits,
        budget,
        soft_budget,
        tpd_limit,
        max_parallel_requests,
        budget_duration,
        budget_extras,
    } = budget_row_or_spend(litellm_budget_table.as_ref(), *spend)?;
    Ok(EndUserRecord {
        end_user_id: Some(user_id.clone()),
        object_permission: object_permission::record(
            object_permission_id.as_deref(),
            object_permission.as_ref(),
        )?,
        found: true,
        blocked: *blocked,
        default_model: default_model.clone(),
        models: models.clone(),
        limits,
        budget,
        soft_budget,
        tpd_limit,
        max_parallel_requests,
        budget_duration,
        budget_extras,
        allowed_model_region: allowed_model_region.map(|region| match region {
            StoredModelRegion::Eu => ModelRegion::Eu,
            StoredModelRegion::Us => ModelRegion::Us,
        }),
    })
}

/// The first column of `end_user` that holds something nothing in the native app reads yet, which
/// sends its request to Python. The record is taken apart by name, so a column added to it does
/// not compile here until it is named as read or listed. Whoever makes the native app honor a
/// column takes it off the list in the same change.
pub(crate) fn unserved(end_user: &EndUserRecord) -> Option<&'static str> {
    let EndUserRecord {
        end_user_id: _,
        object_permission,
        found: _,
        blocked,
        default_model,
        models,
        limits: MinuteLimits {
            rpm: rpm_limit,
            tpm: tpm_limit,
        },
        budget: Budget {
            max: max_budget,
            spend: _,
        },
        soft_budget,
        tpd_limit,
        max_parallel_requests,
        budget_duration,
        allowed_model_region,
        budget_extras,
    } = end_user;
    let object_permission = &object_permission.is_some();
    carried![object_permission]
        .or_else(|| unserved_budget_extras(budget_extras))
        .or_else(|| {
            carried![
                blocked,
                default_model,
                models,
                rpm_limit,
                tpm_limit,
                max_budget,
                soft_budget,
                tpd_limit,
                max_parallel_requests,
                budget_duration,
                allowed_model_region,
            ]
        })
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
        let record = record(&stored(columns)).unwrap();

        assert_eq!(
            (
                record.end_user_id.as_deref(),
                record.found,
                unserved(&record)
            ),
            (Some("eu-1"), true, None)
        );
    }

    #[rstest]
    #[case::a_block(json!({"blocked": true}), "blocked")]
    #[case::a_default_model(json!({"default_model": "claude"}), "default_model")]
    #[case::a_model_list(json!({"models": ["claude"]}), "models")]
    #[case::a_budget(json!({"litellm_budget_table": {"max_budget": 1.0}}), "max_budget")]
    #[case::a_rate_limit_in_its_budget(json!({"litellm_budget_table": {"tpm_limit": 10}}), "tpm_limit")]
    fn a_plain_column_the_row_holds_is_copied_and_named_unserved(
        #[case] columns: Value,
        #[case] column: &str,
    ) {
        assert_eq!(unserved(&record(&stored(columns)).unwrap()), Some(column));
    }

    #[rstest]
    #[case::a_region(json!({"allowed_model_region": "eu"}), Some(ModelRegion::Eu), Some("allowed_model_region"))]
    #[case::null(json!({"allowed_model_region": null}), None, None)]
    fn an_end_users_region_is_read_but_not_enforced(
        #[case] columns: Value,
        #[case] region: Option<ModelRegion>,
        #[case] field: Option<&str>,
    ) {
        let end_user = record(&stored(columns)).unwrap();
        assert_eq!(
            (end_user.allowed_model_region, unserved(&end_user)),
            (region, field)
        );
    }

    #[rstest]
    fn an_end_users_per_model_budget_is_read_but_not_enforced() {
        let entry = stored(json!({"litellm_budget_table": {"model_max_budget": {"claude": 1.0}}}));
        let row = record(&entry).unwrap();
        assert_eq!(
            row.budget_extras.model_max_budget["claude"],
            litellm_gateway_auth_core::record::ModelBudget::Amount(1.0),
        );
        assert_eq!(unserved(&row), Some("model_max_budget"));
    }

    #[rstest]
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
        let row = record(&stored(columns));
        assert_eq!(
            row.as_ref().map_or_else(|field| Some(*field), unserved),
            Some(field)
        );
    }
}
