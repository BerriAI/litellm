#[derive(
    Clone,
    Copy,
    Debug,
    serde::Serialize,
    strum::Display,
    strum::AsRefStr,
    strum::EnumIter,
    strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum TraceTable {
    OtelTraces,
    AgentTracesByKey,
    SpendLogs,
}
