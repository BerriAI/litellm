#[macro_rules_attribute::apply(crate::response_type)]
#[cfg_attr(feature = "schema", schemars(rename = "TraceTableName"))]
#[derive(
    Clone, Copy, Debug, strum::Display, strum::AsRefStr, strum::EnumIter, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
pub enum TraceTable {
    #[strum(serialize = "otel_traces")]
    OtelTraces,
    #[strum(serialize = "agent_traces_by_key")]
    AgentTracesByKey,
    #[strum(serialize = "spend_logs")]
    SpendLogs,
}
