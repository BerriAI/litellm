use crate::Raw;

#[macro_rules_attribute::apply(crate::config_section)]
pub struct McpToolSettings {
    pub mcp_aliases: Option<Raw>,
    pub mcp_semantic_tool_filter: Option<Raw>,
    pub mcp_tool_search: Option<Raw>,
}
