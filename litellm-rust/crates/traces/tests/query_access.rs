use std::collections::BTreeMap;

use litellm_http::Client;
use litellm_traces::{
    Connection, Error, QueryReaders, QueryScope, ensure_schema, query_help, query_sql,
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use testcontainers_modules::{
    clickhouse::ClickHouse,
    testcontainers::{ContainerAsync, ImageExt, runners::AsyncRunner},
};

struct Database {
    _container: ContainerAsync<ClickHouse>,
    client: Client,
    writer: Connection,
    readers: QueryReaders,
}

#[fixture]
async fn database() -> Result<Database, Box<dyn std::error::Error>> {
    let container = ClickHouse::default()
        .with_tag(
            "26.9.6.6@sha256:eb4870e7ca7ed70c259eebfcfbee6cf797017f6b5436c2926bbbfe3d4d28486e",
        )
        .with_env_var("CLICKHOUSE_SKIP_USER_SETUP", "1")
        .start()
        .await?;
    let writer = Connection::parse(&format!(
        "http://{}:{}",
        container.get_host().await?,
        container.get_host_port_ipv4(8123).await?
    ))?;
    let client = Client::no_redirect_for_test();
    ensure_schema(&client, &writer, "trace_test", 7, 7).await?;
    for sql in [
        "INSERT INTO trace_test.otel_traces (TeamId, ApiKeyHash, TraceId, SpanId, Timestamp, SpanAttributes) VALUES ('team-a', 'key-a1', 'shared-trace', 'a1', now(), map('visible', 'a')), ('team-a', 'key-a2', 'shared-trace', 'a2', now(), map('visible', 'a')), ('team-b', 'key-b', 'shared-trace', 'b', now(), map('secret-b', 'b'))",
        "INSERT INTO trace_test.spend_logs (team_id, api_key, request_id, start_time, end_time, metadata) VALUES ('team-a', 'key-a1', 'a1', now(), now(), '{\"visible\":1}'), ('team-a', 'key-a2', 'a2', now(), now(), '{\"visible\":1}'), ('team-b', 'key-b', 'b', now(), now(), '{\"secret_b\":1}')",
        "CREATE TABLE trace_test.private_data (secret String) ENGINE = Memory",
        "INSERT INTO trace_test.private_data VALUES ('hidden')",
    ] {
        let response = client.post(writer.url().clone()).body(sql).send().await?;
        assert!(response.status().is_success(), "{}", response.text().await?);
    }
    let readers = QueryReaders::new(writer.clone(), "trace_test".to_owned());
    Ok(Database {
        _container: container,
        client,
        writer,
        readers,
    })
}

#[rstest]
#[case::team(QueryScope::Team { team_id: "team-a".to_owned() }, vec!["a1", "a2"])]
#[case::project_key(QueryScope::Key { team_id: "team-a".to_owned(), api_key_hash: "key-a1".to_owned() }, vec!["a1"])]
#[case::admin(QueryScope::Admin, vec!["a1", "a2", "b"])]
#[case::quoted_team(QueryScope::Team { team_id: "team-a' OR 1=1 --\\".to_owned() }, vec![])]
#[tokio::test]
async fn queries_and_help_are_scoped_by_the_database(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
    #[case] scope: QueryScope,
    #[case] expected: Vec<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    let reader = database
        .readers
        .connection(&database.client, &scope, "test-master-secret")
        .await?;
    let queries = [
        "SELECT SpanId AS id FROM otel_traces ORDER BY id",
        "SELECT SpanId AS id FROM trace_test.otel_traces WHERE 1 = 1 ORDER BY id",
        "SELECT SpanId AS id FROM merge('trace_test', '^otel_traces$') ORDER BY id",
        "WITH source AS (SELECT * FROM trace_test.otel_traces) SELECT SpanId AS id FROM source ORDER BY id",
        "SELECT SpanId AS id FROM otel_traces UNION DISTINCT SELECT SpanId AS id FROM trace_test.otel_traces ORDER BY id",
        "SELECT t.SpanId AS id FROM otel_traces t INNER JOIN spend_logs s ON t.SpanId = s.request_id ORDER BY id",
        "SELECT request_id AS id FROM spend_logs FINAL ORDER BY id",
    ];
    for sql in queries {
        let body: Value = serde_json::from_str(&query_sql(&database.client, &reader, sql).await?)?;
        assert_eq!(
            body["data"],
            json!(
                expected
                    .iter()
                    .map(|id| json!({"id": id}))
                    .collect::<Vec<_>>()
            ),
            "{sql}"
        );
    }
    let summary: Value = serde_json::from_str(
        &query_sql(
            &database.client,
            &reader,
            "SELECT sum(SpanCount) AS count FROM agent_traces_by_key",
        )
        .await?,
    )?;
    assert_eq!(summary["data"][0]["count"], json!(expected.len()));
    let help = query_help(&database.client, &reader).await?;
    assert_eq!(help.contains("secret_b"), expected.contains(&"b"));
    assert_eq!(help.contains("secret-b"), expected.contains(&"b"));
    let recreated = QueryReaders::new(database.writer.clone(), "trace_test".to_owned());
    let repeated = recreated
        .connection(&database.client, &scope, "test-master-secret")
        .await?;
    assert_eq!(reader.url(), repeated.url());
    Ok(())
}

#[rstest]
#[tokio::test]
async fn rotating_master_secret_revokes_previous_reader_credentials(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    let scope = QueryScope::Team {
        team_id: "team-a".to_owned(),
    };
    let old_reader = database
        .readers
        .connection(&database.client, &scope, "old-master-secret")
        .await?;
    let old_result = query_sql(
        &database.client,
        &old_reader,
        "SELECT SpanId AS id FROM otel_traces ORDER BY id",
    )
    .await?;
    let old_rows: Value = serde_json::from_str(&old_result)?;
    assert_eq!(old_rows["data"], json!([{ "id": "a1" }, { "id": "a2" }]));

    let rotated_readers = QueryReaders::new(database.writer.clone(), "trace_test".into());
    let new_reader = rotated_readers
        .connection(&database.client, &scope, "new-master-secret")
        .await?;
    assert!(
        query_sql(
            &database.client,
            &old_reader,
            "SELECT SpanId AS id FROM otel_traces ORDER BY id",
        )
        .await
        .is_err()
    );
    let new_result = query_sql(
        &database.client,
        &new_reader,
        "SELECT SpanId AS id FROM otel_traces ORDER BY id",
    )
    .await?;
    let new_rows: Value = serde_json::from_str(&new_result)?;
    assert_eq!(new_rows["data"], json!([{ "id": "a1" }, { "id": "a2" }]));
    assert_eq!(old_reader.url().username(), new_reader.url().username());
    assert_ne!(old_reader.url().password(), new_reader.url().password());
    Ok(())
}

#[rstest]
#[tokio::test]
async fn managed_reader_rejects_privilege_and_scope_bypasses(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    let scope = QueryScope::Team {
        team_id: "team-a".to_owned(),
    };
    let reader = database
        .readers
        .connection(&database.client, &scope, "test-master-secret")
        .await?;
    for sql in [
        "INSERT INTO otel_traces (TraceId) VALUES ('injected')",
        "DROP TABLE otel_traces",
        "SELECT * FROM private_data",
        "SELECT * FROM otel_traces SETTINGS readonly = 0",
        "SELECT * FROM otel_traces SETTINGS max_memory_usage = 0",
        "SELECT * FROM otel_traces SETTINGS max_execution_time = 0",
        "CREATE USER scope_bypass",
        "CREATE NAMED COLLECTION scope_bypass AS host = 'localhost'",
        "BACKUP TABLE otel_traces TO Disk('default', 'scope-bypass')",
        "SELECT * FROM url('http://127.0.0.1:1/', 'LineAsString', 'line String')",
        "SELECT * FROM remote('127.0.0.1', 'trace_test', 'otel_traces')",
    ] {
        assert!(
            matches!(
                query_sql(&database.client, &reader, sql).await,
                Err(Error::QueryFailed(_))
            ),
            "{sql}"
        );
    }
    let roles: Value = serde_json::from_str(
        &query_sql(&database.client, &reader, "SELECT enabledRoles() AS roles").await?,
    )?;
    assert_eq!(roles["data"], json!([{ "roles": [] }]));
    let rows: Value = serde_json::from_str(
        &query_sql(
            &database.client,
            &reader,
            "SELECT DISTINCT TeamId FROM otel_traces",
        )
        .await?,
    )?;
    assert_eq!(rows["data"], json!([{ "TeamId": "team-a" }]));
    Ok(())
}

#[rstest]
#[tokio::test]
async fn provisioning_failure_never_returns_a_writer_connection(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    let reader = database
        .readers
        .connection(&database.client, &QueryScope::Admin, "test-master-secret")
        .await?;
    let no_provision_privileges = QueryReaders::new(reader, "trace_test".to_owned());
    let result = no_provision_privileges
        .connection(
            &database.client,
            &QueryScope::Team {
                team_id: "team-a".to_owned(),
            },
            "other-secret",
        )
        .await;
    assert!(result.is_err());
    assert!(
        database
            .readers
            .connection(&database.client, &QueryScope::Admin, "")
            .await
            .is_err()
    );
    assert!(
        database
            .readers
            .connection(
                &database.client,
                &QueryScope::Team {
                    team_id: String::new()
                },
                "test-master-secret"
            )
            .await
            .is_err()
    );
    let permits = (0..8)
        .map(|_| database.readers.acquire())
        .collect::<Result<Vec<_>, _>>()?;
    assert!(database.readers.acquire().is_err());
    drop(permits);
    assert!(database.readers.acquire().is_ok());
    let rows = litellm_traces::execute_read(
        &database.client,
        &database.writer,
        "SELECT count() AS count FROM trace_test.otel_traces",
        &BTreeMap::new(),
    )
    .await?;
    let rows: Value = serde_json::from_str(&rows)?;
    assert_eq!(rows["data"][0]["count"], 3);
    Ok(())
}
