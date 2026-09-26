#![cfg(feature = "postgres")]

use litellm_db::queries::autorouter::{
    HistoryCheckpoint, LockedComparison, advance_baseline_comparison_revision,
    create_baseline_comparison, find_baseline_observation_without_spend_log,
    insert_baseline_observation, lock_baseline_comparison, publish_baseline_comparison_history,
};
use litellm_db_testing::MigratedPostgres;
use rstest::{fixture, rstest};
use sqlx::{PgConnection, Postgres, pool::PoolConnection};

const SCOPE: &str = "key:session:router";

#[fixture]
async fn database() -> MigratedPostgres {
    MigratedPostgres::start().await.unwrap()
}

async fn connection(database: &MigratedPostgres) -> PoolConnection<Postgres> {
    database.pool().acquire().await.unwrap()
}

async fn create(conn: &mut PgConnection) -> bool {
    create_baseline_comparison(conn, SCOPE, "key", "session", "router")
        .await
        .unwrap()
}

#[rstest]
#[tokio::test]
async fn an_unknown_scope_has_nothing_to_lock_or_write(#[future(awt)] database: MigratedPostgres) {
    let mut conn = connection(&database).await;

    assert_eq!(
        lock_baseline_comparison(&mut conn, SCOPE).await.unwrap(),
        None
    );
    assert!(
        !advance_baseline_comparison_revision(&mut conn, SCOPE, 1)
            .await
            .unwrap()
    );
    assert!(
        !publish_baseline_comparison_history(&mut conn, SCOPE, "{}")
            .await
            .unwrap()
    );
}

#[rstest]
#[tokio::test]
async fn a_scope_is_created_once(#[future(awt)] database: MigratedPostgres) {
    let mut conn = connection(&database).await;

    assert!(create(&mut conn).await);
    assert!(!create(&mut conn).await);
}

#[rstest]
#[case::without_a_session(false, true)]
#[case::with_a_session(true, false)]
#[tokio::test]
async fn a_new_comparison_starts_from_its_initial_checkpoint(
    #[future(awt)] database: MigratedPostgres,
    #[case] session_exists: bool,
    #[case] equivalent: bool,
) {
    let mut conn = connection(&database).await;
    if session_exists {
        sqlx::query!(
            r#"INSERT INTO "LiteLLM_AutoRouterSession"
                (api_key, session_id, router_name, router_type, first_turn_at, last_turn_at, last_model)
            VALUES ('key', 'session', 'router', 'auto', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'model')"#
        )
        .execute(&mut *conn)
        .await
        .unwrap();
    }
    create(&mut conn).await;

    let locked = lock_baseline_comparison(&mut conn, SCOPE).await.unwrap();

    assert_eq!(
        locked,
        Some(LockedComparison {
            revision: 0,
            published_revision: 0,
            retired: false,
            checkpoint: HistoryCheckpoint::Initial { equivalent },
        })
    );
}

#[rstest]
#[tokio::test]
async fn the_lock_reads_the_published_checkpoint_behind_newer_revisions(
    #[future(awt)] database: MigratedPostgres,
) {
    let mut conn = connection(&database).await;
    create(&mut conn).await;
    advance_baseline_comparison_revision(&mut conn, SCOPE, 3)
        .await
        .unwrap();
    assert!(
        publish_baseline_comparison_history(&mut conn, SCOPE, r#"{"equivalent":false}"#)
            .await
            .unwrap()
    );
    assert!(
        advance_baseline_comparison_revision(&mut conn, SCOPE, 5)
            .await
            .unwrap()
    );
    sqlx::query!(
        r#"UPDATE "LiteLLM_AutoRouterBaselineComparison" SET retired = TRUE WHERE scope = $1"#,
        SCOPE
    )
    .execute(&mut *conn)
    .await
    .unwrap();

    let locked = lock_baseline_comparison(&mut conn, SCOPE).await.unwrap();

    assert_eq!(
        locked,
        Some(LockedComparison {
            revision: 5,
            published_revision: 3,
            retired: true,
            checkpoint: HistoryCheckpoint::Published {
                history: r#"{"equivalent":false}"#.to_owned(),
            },
        })
    );
}

#[rstest]
#[tokio::test]
async fn an_observation_is_inserted_once_and_counts_as_missing_its_spend_log(
    #[future(awt)] database: MigratedPostgres,
) {
    let mut conn = connection(&database).await;
    assert!(
        !find_baseline_observation_without_spend_log(&mut conn, SCOPE)
            .await
            .unwrap()
    );

    assert!(
        insert_baseline_observation(&mut conn, "request", SCOPE, 1.5, 1, "{}")
            .await
            .unwrap()
    );
    assert!(
        !insert_baseline_observation(&mut conn, "request", SCOPE, 1.5, 1, "{}")
            .await
            .unwrap()
    );

    assert!(
        find_baseline_observation_without_spend_log(&mut conn, SCOPE)
            .await
            .unwrap()
    );
}
