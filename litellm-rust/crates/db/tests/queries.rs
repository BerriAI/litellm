#![cfg(feature = "postgres-tests")]

mod support;

use std::collections::BTreeSet;

use rstest::rstest;
use sqlx::{Column, Executor, Postgres, SqlSafeStr, Transaction};
use support::transaction;

const KEY_AUTH_COMBINED_VIEW: &str =
    include_str!("../../../../litellm/proxy/db/queries/keys/key_auth_combined_view.sql");

#[rstest]
#[tokio::test]
async fn key_auth_selects_every_verification_token_column(
    #[future(awt)]
    #[from(transaction)]
    mut transaction: Transaction<'static, Postgres>,
) {
    let table_columns: BTreeSet<String> = sqlx::query_scalar!(
        r#"SELECT column_name AS "column_name!" FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'LiteLLM_VerificationToken'"#
    )
    .fetch_all(&mut *transaction)
    .await
    .unwrap()
    .into_iter()
    .collect();
    let selected: BTreeSet<String> = (&mut *transaction)
        .describe(KEY_AUTH_COMBINED_VIEW.into_sql_str())
        .await
        .unwrap()
        .columns()
        .iter()
        .map(|column| column.name().to_owned())
        .collect();

    let missing: Vec<&String> = table_columns.difference(&selected).collect();

    assert!(!table_columns.is_empty());
    assert_eq!(missing, Vec::<&String>::new());
}
