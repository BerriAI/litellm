#[macro_rules_attribute::apply(response_type)]
#[cfg_attr(feature = "schema", schemars(rename = "TraceTableName"))]
#[derive(
    Clone, Copy, Debug, strum::Display, strum::AsRefStr, strum::EnumIter, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum TraceTable {
    OtelTraces,
    AgentTracesByKey,
    SpendLogs,
}
