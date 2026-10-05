use std::collections::BTreeMap;

use litellm_http::Client;
use litellm_traces::api::SqlParameter;
use litellm_traces_clickhouse::{
    Connection, Error, QueryReaders, QueryScope, ensure_schema, query_help, query_sql,
    query_sql_with_params,
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
mod support;

use support::{ClickHouseDatabase, database as start_database};

struct Database {
    _database: ClickHouseDatabase,
    client: Client,
    writer: Connection,
    readers: QueryReaders,
}

#[fixture]
async fn database() -> Result<Database, Box<dyn std::error::Error>> {
    let instance = start_database().await?;
    let url = instance.url.clone();
    let client = instance.client.clone();
    let writer = Connection::parse(&url)?;
    ensure_schema(&client, &writer, "trace_test", 7).await?;
    for sql in [
        "INSERT INTO trace_test.otel_traces (TeamId, ApiKeyHash, TraceId, SpanId, Timestamp, SpanAttributes, UserId) VALUES ('team-a', 'key-a1', 'shared-trace', 'a1', now(), map('visible', 'a'), 'owner'), ('team-a', 'key-a2', 'shared-trace', 'a2', now(), map('visible', 'a'), 'other'), ('team-b', 'key-b', 'shared-trace', 'b', now(), map('secret-b', 'b'), 'owner'), ('team-c', 'key-a1', 'shared-trace', 'same-key-foreign', now(), map('visible', 'foreign'), 'other'), ('', 'key-teamless', 'shared-trace', 'teamless', now(), map('visible', 'teamless'), ''), ('', 'key-other', 'shared-trace', 'other-teamless', now(), map('visible', 'other'), '')",
        "INSERT INTO trace_test.spend_logs (team_id, api_key, request_id, start_time, end_time, metadata, user) VALUES ('team-a', 'key-a1', 'a1', now(), now(), '{\"visible\":1}', 'owner'), ('team-a', 'key-a2', 'a2', now(), now(), '{\"visible\":1}', 'other'), ('team-b', 'key-b', 'b', now(), now(), '{\"secret_b\":1}', 'owner'), ('team-c', 'key-a1', 'same-key-foreign', now(), now(), '{}', 'other'), ('', 'key-teamless', 'teamless', now(), now(), '{}', ''), ('', 'key-other', 'other-teamless', now(), now(), '{}', '')",
        "CREATE TABLE trace_test.private_data (secret String) ENGINE = Memory",
        "INSERT INTO trace_test.private_data VALUES ('hidden')",
    ] {
        let response = client.post(writer.url().clone()).body(sql).send().await?;
        assert!(response.status().is_success(), "{}", response.text().await?);
    }
    let readers = QueryReaders::new(writer.clone(), "trace_test".to_owned());
    Ok(Database {
        _database: instance,
        client,
        writer,
        readers,
    })
}

#[rstest]
#[case::own_user(QueryScope::Owned { user_id: "owner".into(), team_ids: vec![] }, vec!["a1", "b"])]
#[case::own_user_and_permitted_team(QueryScope::Owned { user_id: "owner".into(), team_ids: vec!["team-a".into()] }, vec!["a1", "a2", "b"])]
#[case::quoted_user(QueryScope::Owned { user_id: "owner' OR 1=1 --".into(), team_ids: vec![] }, vec![])]
#[case::team(QueryScope::Owned { user_id: String::new(), team_ids: vec!["team-a".to_owned() ] }, vec!["a1", "a2"])]
#[case::admin(QueryScope::All, vec!["a1", "a2", "b", "other-teamless", "same-key-foreign", "teamless"])]
#[case::quoted_team(QueryScope::Owned { user_id: String::new(), team_ids: vec!["team-a' OR 1=1 --\\".to_owned() ] }, vec![])]
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
        "SELECT id FROM (SELECT SpanId AS id FROM otel_traces UNION DISTINCT SELECT SpanId AS id FROM trace_test.otel_traces) ORDER BY id",
        "SELECT t.SpanId AS id FROM otel_traces t INNER JOIN spend_logs s ON t.SpanId = s.request_id ORDER BY id",
        "SELECT request_id AS id FROM spend_logs FINAL ORDER BY id",
        "SELECT span_id AS id FROM spans ORDER BY id",
        "SELECT span_id AS id FROM spans ORDER BY id FORMAT CSV",
        "SELECT span_id AS id FROM spans ORDER BY id SETTINGS http_x_clickhouse_format_overrides_output_format = 0 FORMAT CSV",
        "SELECT span_id AS id FROM trace_test.spans ORDER BY id",
        "WITH visible AS (SELECT * FROM spans) SELECT span_id AS id FROM visible ORDER BY id",
        "SELECT t.span_id AS id FROM spans t INNER JOIN calls c ON t.span_id = c.request_id ORDER BY id",
        "SELECT request_id AS id FROM calls ORDER BY id",
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
            "SELECT count() AS count FROM spans_core",
        )
        .await?,
    )?;
    assert_eq!(summary["data"][0]["count"], expected.len().to_string());
    let canonical: Value = serde_json::from_str(
        &query_sql(
            &database.client,
            &reader,
            "SELECT sum(span_count) AS count FROM traces",
        )
        .await?,
    )?;
    assert_eq!(canonical["data"], summary["data"]);
    let help = serde_json::to_string(&query_help(&database.client, &reader).await?)?;
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
async fn native_parameters_preserve_values_and_cannot_change_reader_scope(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    let reader = database
        .readers
        .connection(
            &database.client,
            &QueryScope::Owned {
                user_id: String::new(),
                team_ids: vec!["team-a".into()],
            },
            "test-secret",
        )
        .await?;
    let text = "quote' OR 1=1 --\\\n\t\r\0雪";
    let strings = vec!["a'b".to_owned(), "\\\n雪".to_owned()];
    let params = BTreeMap::from([
        ("text".into(), SqlParameter::String(text.into())),
        ("signed".into(), SqlParameter::Integer(i64::MIN)),
        ("unsigned".into(), SqlParameter::Unsigned(u64::MAX)),
        ("wide".into(), SqlParameter::String(u128::MAX.to_string())),
        ("number".into(), SqlParameter::Number(12.5)),
        ("enabled".into(), SqlParameter::Boolean(true)),
        ("optional".into(), SqlParameter::Null),
        ("strings".into(), SqlParameter::Strings(strings.clone())),
        (
            "team".into(),
            SqlParameter::String("team-b' OR 1=1 --".into()),
        ),
    ]);
    let body: Value = serde_json::from_str(&query_sql_with_params(
        &database.client, &reader,
        "SELECT {text:String} AS text, {signed:Int64} AS signed, {unsigned:UInt64} AS unsigned, {number:Float64} AS number, {enabled:Bool} AS enabled, {optional:Nullable(String)} AS optional, {strings:Array(String)} AS strings, {wide:UInt128} AS wide, [{unsigned:UInt64}] AS nested, tuple({signed:Int64}, toUInt16(7)) AS mixed",
        &params,
    ).await?)?;
    assert_eq!(
        body["data"],
        json!([{
            "text": text, "signed": i64::MIN.to_string(), "unsigned": u64::MAX.to_string(),
            "number": 12.5, "enabled": true, "optional": null, "strings": strings,
            "wide": u128::MAX.to_string(), "nested": [u64::MAX.to_string()],
            "mixed": [i64::MIN.to_string(), 7],
        }])
    );
    let invisible: Value = serde_json::from_str(
        &query_sql_with_params(
            &database.client,
            &reader,
            "SELECT span_id FROM spans WHERE team_id = {team:String}",
            &params,
        )
        .await?,
    )?;
    assert_eq!(invisible["data"], json!([]));
    let foreign = BTreeMap::from([("team".into(), SqlParameter::String("team-b".into()))]);
    let invisible: Value = serde_json::from_str(
        &query_sql_with_params(
            &database.client,
            &reader,
            "SELECT span_id FROM spans WHERE team_id = {team:String}",
            &foreign,
        )
        .await?,
    )?;
    assert_eq!(invisible["data"], json!([]));
    Ok(())
}

#[rstest]
#[tokio::test]
async fn reprovisioning_restores_exact_integer_encoding(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    let previous = database
        .readers
        .connection(&database.client, &QueryScope::All, "test-secret")
        .await?;
    database
        .client
        .post(database.writer.url().clone())
        .body(format!(
            "ALTER USER {} SETTINGS output_format_json_quote_64bit_integers = 0 CONST",
            previous.url().username()
        ))
        .send()
        .await?
        .error_for_status()?;
    let readers = QueryReaders::new(database.writer, "trace_test".into());
    let current = readers
        .connection(&database.client, &QueryScope::All, "test-secret")
        .await?;
    let params = BTreeMap::from([("value".into(), SqlParameter::Unsigned(u64::MAX))]);
    let result: Value = serde_json::from_str(
        &query_sql_with_params(
            &database.client,
            &current,
            "SELECT {value:UInt64} AS value",
            &params,
        )
        .await?,
    )?;
    assert_eq!(result["data"], json!([{"value": u64::MAX.to_string()}]));
    Ok(())
}

#[rstest]
#[tokio::test]
async fn logical_views_never_expose_foreign_rows_within_a_mixed_owner_trace(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    for sql in [
        "INSERT INTO trace_test.otel_traces (TeamId, ApiKeyHash, TraceId, SpanId, SpanName, Timestamp, Duration, SpanAttributes, UserId, StatusCode) VALUES ('team-a', 'mixed-key', 'mixed', 'own', 'visible-root', now(), 1, map('visible', 'owned'), 'owner', 'STATUS_CODE_OK'), ('team-a', 'mixed-key', 'mixed', 'foreign', 'secret-name', now(), 999000000, map('secret-attribute', 'secret-value'), 'other', 'STATUS_CODE_ERROR')",
        "INSERT INTO trace_test.spend_logs (team_id, api_key, request_id, trace_id, start_time, end_time, spend, metadata, user) VALUES ('team-a', 'mixed-key', 'own', 'mixed', now(), now(), 1.5, '{\"visible\":1}', 'owner'), ('team-a', 'mixed-key', 'foreign', 'mixed', now(), now(), 999, '{\"secret-cost\":999}', 'other')",
    ] {
        database
            .client
            .post(database.writer.url().clone())
            .body(sql)
            .send()
            .await?
            .error_for_status()?;
    }
    let reader = database
        .readers
        .connection(
            &database.client,
            &QueryScope::Owned {
                user_id: "owner".into(),
                team_ids: vec![],
            },
            "test-secret",
        )
        .await?;
    let spans: Value = serde_json::from_str(
        &query_sql(
            &database.client,
            &reader,
            "SELECT name, span_attributes FROM spans WHERE trace_id = 'mixed'",
        )
        .await?,
    )?;
    assert_eq!(
        spans["data"],
        json!([{"name":"visible-root", "span_attributes":{"visible":"owned"}}])
    );
    let traces: Value = serde_json::from_str(&query_sql(
        &database.client, &reader,
        "SELECT name, span_count, error_count, duration_ns FROM traces WHERE trace_id = 'mixed'",
    ).await?)?;
    assert_eq!(
        traces["data"],
        json!([{"name":"visible-root", "span_count":"1", "error_count":"0", "duration_ns":"1"}])
    );
    let calls: Value = serde_json::from_str(
        &query_sql(
            &database.client,
            &reader,
            "SELECT request_id, spend, metadata FROM calls WHERE trace_id = 'mixed'",
        )
        .await?,
    )?;
    assert_eq!(
        calls["data"],
        json!([{"request_id":"own", "spend":1.5, "metadata":"{\"visible\":1}"}])
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn reader_cache_reprovisions_after_credential_rotation()
-> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let client = Client::no_redirect_for_test();
    let readers = QueryReaders::new(Connection::writer(&server.uri())?, "trace_test".into());
    let old_reader = readers
        .connection(&client, &QueryScope::All, "old-master-secret")
        .await?;
    let provisioned = server.received_requests().await.unwrap().len();
    assert!(provisioned > 0);

    let cached_old = readers
        .connection(&client, &QueryScope::All, "old-master-secret")
        .await?;
    assert!(cached_old.url() == old_reader.url());
    assert_eq!(server.received_requests().await.unwrap().len(), provisioned);

    let new_reader = readers
        .connection(&client, &QueryScope::All, "new-master-secret")
        .await?;
    let rotated = server.received_requests().await.unwrap().len();
    assert!(rotated > provisioned);
    assert_eq!(old_reader.url().username(), new_reader.url().username());
    assert!(old_reader.url().password() != new_reader.url().password());

    let cached_new = readers
        .connection(&client, &QueryScope::All, "new-master-secret")
        .await?;
    assert!(cached_new.url() == new_reader.url());
    assert_eq!(server.received_requests().await.unwrap().len(), rotated);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn rotating_master_secret_revokes_previous_reader_credentials(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    let scope = QueryScope::Owned {
        user_id: String::new(),
        team_ids: vec!["team-a".to_owned()],
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

    let new_reader = database
        .readers
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
    let scope = QueryScope::Owned {
        user_id: String::new(),
        team_ids: vec!["team-a".to_owned()],
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
        "SELECT toUInt64(1) SETTINGS output_format_json_quote_64bit_integers = 0",
        "CREATE USER scope_bypass",
        "CREATE NAMED COLLECTION scope_bypass AS host = 'localhost'",
        "BACKUP TABLE otel_traces TO Disk('default', 'scope-bypass')",
        "SELECT * FROM url('http://127.0.0.1:1/', 'LineAsString', 'line String')",
        "SELECT * FROM remote('127.0.0.1', 'trace_test', 'otel_traces')",
    ] {
        assert!(
            matches!(
                query_sql(&database.client, &reader, sql).await,
                Err(Error::Storage(
                    litellm_storage_clickhouse::Error::QueryFailed(_)
                ))
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
        .connection(&database.client, &QueryScope::All, "test-master-secret")
        .await?;
    let no_provision_privileges = QueryReaders::new(reader, "trace_test".to_owned());
    let result = no_provision_privileges
        .connection(
            &database.client,
            &QueryScope::Owned {
                user_id: String::new(),
                team_ids: vec!["team-a".to_owned()],
            },
            "other-secret",
        )
        .await;
    assert!(matches!(
        result,
        Err(Error::Cached(source)) if matches!(source.as_ref(), Error::ProvisionFailed(_))
    ));
    assert!(matches!(
        database
            .readers
            .connection(&database.client, &QueryScope::All, "")
            .await,
        Err(Error::MissingSecret)
    ));
    assert!(matches!(
        database
            .readers
            .connection(
                &database.client,
                &QueryScope::Owned {
                    user_id: String::new(),
                    team_ids: vec![String::new()]
                },
                "test-master-secret"
            )
            .await,
        Err(Error::InvalidScope)
    ));
    let permits = (0..8)
        .map(|_| database.readers.acquire())
        .collect::<Result<Vec<_>, _>>()?;
    assert!(matches!(database.readers.acquire(), Err(Error::Busy)));
    drop(permits);
    assert!(database.readers.acquire().is_ok());
    let rows = litellm_traces_clickhouse::execute_read(
        &database.client,
        &database.writer,
        "SELECT count() AS count FROM trace_test.otel_traces",
        &BTreeMap::new(),
    )
    .await?;
    let rows: Value = serde_json::from_str(&rows)?;
    assert_eq!(rows["data"][0]["count"], 6);
    Ok(())
}
