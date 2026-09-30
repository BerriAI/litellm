use std::time::Duration;

use litellm_http::Client;

use crate::Connection;
use crate::Error;

const SCHEMA_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

const MIGRATIONS: [&str; 4] = [
    include_str!("../migrations/0001_otel_traces.sql"),
    include_str!("../migrations/0002_agent_traces.sql"),
    include_str!("../migrations/0003_agent_traces_mv.sql"),
    include_str!("../migrations/0004_spend_logs.sql"),
];

pub fn schema_statements(
    database: &str,
    trace_retention_days: u32,
    spend_log_retention_days: u32,
) -> Result<Vec<String>, Error> {
    if database.is_empty()
        || !database
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_')
        || trace_retention_days == 0
        || spend_log_retention_days == 0
    {
        return Err(Error::InvalidSchema);
    }
    let database = format!("`{database}`");
    Ok(
        std::iter::once(format!("CREATE DATABASE IF NOT EXISTS {database}"))
            .chain(MIGRATIONS.iter().map(|sql| {
                sql.replace("{database}", &database)
                    .replace("{trace_retention_days}", &trace_retention_days.to_string())
                    .replace(
                        "{spend_log_retention_days}",
                        &spend_log_retention_days.to_string(),
                    )
            }))
            .collect(),
    )
}

pub async fn ensure_schema(
    client: &Client,
    connection: &Connection,
    database: &str,
    trace_retention_days: u32,
    spend_log_retention_days: u32,
) -> Result<(), Error> {
    ensure_schema_with_timeout(
        client,
        connection,
        database,
        trace_retention_days,
        spend_log_retention_days,
        SCHEMA_REQUEST_TIMEOUT,
    )
    .await
}

async fn ensure_schema_with_timeout(
    client: &Client,
    connection: &Connection,
    database: &str,
    trace_retention_days: u32,
    spend_log_retention_days: u32,
    request_timeout: Duration,
) -> Result<(), Error> {
    for statement in schema_statements(database, trace_retention_days, spend_log_retention_days)? {
        let response = client
            .post(connection.url().clone())
            .timeout(request_timeout)
            .body(statement)
            .send()
            .await
            .map_err(|_| Error::Transport)?;
        if !response.status().is_success() {
            return Err(Error::SchemaFailed(response.status().as_u16()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use tokio::net::TcpListener;

    use super::*;

    #[tokio::test]
    async fn ensure_schema_times_out_when_server_never_answers() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (_socket, _peer) = listener.accept().await.unwrap();
            std::future::pending::<()>().await;
        });
        let client = Client::no_redirect_for_test();
        let connection = Connection::writer(&url).unwrap();

        let started = Instant::now();
        let result = ensure_schema_with_timeout(
            &client,
            &connection,
            "trace_test",
            7,
            14,
            Duration::from_millis(200),
        )
        .await;
        let elapsed = started.elapsed();

        assert!(matches!(result, Err(Error::Transport)), "{result:?}");
        assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
        server.abort();
    }
}
