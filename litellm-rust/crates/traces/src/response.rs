#[macro_rules_attribute::apply(crate::response_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceSQLResponse {
    #[cfg_attr(
        feature = "schema",
        schemars(extend("x-python-normalized" = {"type": "tuple[Mapping[str, JsonValue], ...]"}))
    )]
    pub data: Vec<serde_json::Map<String, serde_json::Value>>,
}
