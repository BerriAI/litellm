#[derive(
    Clone, Copy, Debug, strum::Display, strum::AsRefStr, strum::EnumIter, strum::IntoStaticStr,
)]
#[strum(serialize_all = "snake_case")]
pub enum TraceTable {
    OtelTraces,
    AgentTracesByKey,
    SpendLogs,
}
