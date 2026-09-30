use crate::Error;

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
