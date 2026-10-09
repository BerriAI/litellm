pub const TRACE_PAGE_SIZE_MIN: u16 = 1;
pub const TRACE_PAGE_SIZE_MAX: u16 = 500;

#[macro_rules_attribute::apply(crate::request_type)]
#[derive(Clone, Debug)]
pub struct TraceListRequest {
    /// Window start, unix ms. Default: 24h ago
    #[serde(default)]
    pub start_ms: Option<i64>,
    /// Window end, unix ms. Default: now
    #[serde(default)]
    pub end_ms: Option<i64>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 512)))]
    pub cursor: Option<String>,
}

#[macro_rules_attribute::apply(crate::request_type)]
#[derive(Clone, Debug)]
pub struct TraceDetailRequest {
    #[serde(default)]
    pub trace_ref: String,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 512)))]
    pub cursor: Option<String>,
    #[serde(default)]
    #[cfg_attr(
        feature = "schema",
        schemars(range(min = TRACE_PAGE_SIZE_MIN, max = TRACE_PAGE_SIZE_MAX))
    )]
    pub page_size: Option<u16>,
}

#[macro_rules_attribute::apply(crate::request_type)]
#[derive(Clone, Debug)]
pub struct TraceSpanRequest {
    #[serde(default)]
    pub trace_ref: String,
}

#[macro_rules_attribute::apply(crate::request_type)]
#[derive(Clone, Debug)]
pub struct TraceErrorPageRequest {
    #[serde(default)]
    pub trace_ref: String,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 512)))]
    pub cursor: Option<String>,
}

#[macro_rules_attribute::apply(crate::request_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceQueryRequest {
    pub sql: String,
}
