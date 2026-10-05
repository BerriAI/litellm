#[macro_rules_attribute::apply(request_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceListRequest {
    #[serde(default)]
    pub start_ms: Option<i64>,
    #[serde(default)]
    pub end_ms: Option<i64>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 512)))]
    pub cursor: Option<String>,
}

#[macro_rules_attribute::apply(request_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceDetailRequest {
    #[serde(default)]
    pub trace_ref: String,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 512)))]
    pub cursor: Option<String>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 500)))]
    pub page_size: Option<u16>,
}

#[macro_rules_attribute::apply(request_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceSpanRequest {
    #[serde(default)]
    pub trace_ref: String,
}

#[macro_rules_attribute::apply(request_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceErrorPageRequest {
    #[serde(default)]
    pub trace_ref: String,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 512)))]
    pub cursor: Option<String>,
}

#[macro_rules_attribute::apply(request_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceQueryRequest {
    pub sql: String,
}
