//! The end user a request names, read from the proxy's database on every request that names one.
//!
//! With a database the proxy applies an end user's block, budget, limits, region, default model,
//! model list and object permission to any caller. The native app honors none of them, so it only
//! answers for an end user that has no row, or whose row holds none of them. The row is read each
//! time and never kept: this process hears none of the evictions the proxy's customer endpoints
//! send, so a kept set would go on serving an end user blocked after it was read.

use futures_util::future::BoxFuture;
use litellm_error::Source;
use litellm_repositories::{EndUserRecord, EndUserRepository};

/// An end user whose row the database did not answer for.
pub(crate) const UNREAD: &str = "unread";

/// The proxy's end users by id.
pub trait EndUserLookup: Send + Sync {
    /// The row of `end_user_id`, or `None` when the proxy holds none.
    fn find<'a>(
        &'a self,
        end_user_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<EndUserRecord>, Source>>;
}

impl EndUserLookup for EndUserRepository {
    fn find<'a>(
        &'a self,
        end_user_id: &'a str,
    ) -> BoxFuture<'a, Result<Option<EndUserRecord>, Source>> {
        Box::pin(async move {
            self.find_with_budget_and_object_permission(end_user_id)
                .await
                .map_err(Source::new)
        })
    }
}

/// The first column of `end_user` that holds something the native app does not honor, the same
/// columns the proxy's registry of restricting end users is selected by. The budget and the
/// object permission are only joined to a row that names one.
fn carried(end_user: &EndUserRecord) -> Option<&'static str> {
    let EndUserRecord {
        user_id: _,
        alias: _,
        spend: _,
        litellm_budget_table: _,
        object_permission: _,
        blocked,
        budget_id,
        allowed_model_region,
        default_model,
        models,
        object_permission_id,
    } = end_user;
    [
        ("blocked", *blocked),
        ("budget_id", budget_id.is_some()),
        ("allowed_model_region", allowed_model_region.is_some()),
        ("default_model", default_model.is_some()),
        ("models", !models.is_empty()),
        ("object_permission_id", object_permission_id.is_some()),
    ]
    .into_iter()
    .find(|(_, held)| *held)
    .map(|(column, _)| column)
}

/// What keeps the native app from answering for `end_user_id`: the column of its row that holds
/// something, or that its row could not be read. `None` is an end user the proxy applies nothing
/// for.
pub(crate) async fn unhonored(
    end_users: &dyn EndUserLookup,
    end_user_id: &str,
) -> Option<&'static str> {
    match end_users.find(end_user_id).await {
        Ok(found) => found.as_ref().and_then(carried),
        Err(_) => Some(UNREAD),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// The row the proxy writes for an end user it has only tracked the spend of.
    fn plain() -> EndUserRecord {
        EndUserRecord {
            user_id: "customer".to_owned(),
            alias: None,
            spend: 0.0,
            allowed_model_region: None,
            default_model: None,
            models: vec![],
            budget_id: None,
            litellm_budget_table: None,
            object_permission_id: None,
            object_permission: None,
            blocked: false,
        }
    }

    #[rstest]
    #[case::as_tracked(|_: &mut EndUserRecord| {})]
    #[case::named_and_spent(|end_user: &mut EndUserRecord| {
        end_user.alias = Some("acme".to_owned());
        end_user.spend = 4.2;
    })]
    fn a_row_that_restricts_nothing_holds_nothing(#[case] change: fn(&mut EndUserRecord)) {
        let mut end_user = plain();
        change(&mut end_user);

        assert_eq!(carried(&end_user), None);
    }

    #[rstest]
    #[case::blocked(|end_user: &mut EndUserRecord| end_user.blocked = true, "blocked")]
    #[case::budget_id(
        |end_user: &mut EndUserRecord| end_user.budget_id = Some("budget".to_owned()),
        "budget_id"
    )]
    #[case::allowed_model_region(
        |end_user: &mut EndUserRecord| end_user.allowed_model_region = Some("eu".to_owned()),
        "allowed_model_region"
    )]
    #[case::default_model(
        |end_user: &mut EndUserRecord| end_user.default_model = Some("claude".to_owned()),
        "default_model"
    )]
    #[case::models(
        |end_user: &mut EndUserRecord| end_user.models = vec!["claude".to_owned()],
        "models"
    )]
    #[case::object_permission_id(
        |end_user: &mut EndUserRecord| end_user.object_permission_id = Some("permission".to_owned()),
        "object_permission_id"
    )]
    fn a_row_that_holds_anything_the_proxy_applies_is_named_by_the_column(
        #[case] change: fn(&mut EndUserRecord),
        #[case] column: &str,
    ) {
        let mut end_user = plain();
        change(&mut end_user);

        assert_eq!(carried(&end_user), Some(column));
    }
}
