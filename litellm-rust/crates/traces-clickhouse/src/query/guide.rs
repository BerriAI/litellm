use askama::Template;
use serde::Serialize;

use super::{AttributeCatalog, MetadataCatalog, TableSchema};
use crate::{Error, NormalizedFieldDefinition};

#[derive(Template)]
#[template(path = "query_help.jinja", escape = "none", blocks = [
    "recent_spans_name",
    "recent_spans_sql",
    "custom_metadata_name",
    "custom_metadata_sql",
    "nested_metadata_name",
    "nested_metadata_sql",
    "correlated_calls_name",
    "correlated_calls_sql",
    "discover_keys_name",
    "discover_keys_sql",
    "time_window",
    "reader_limits",
    "reader_profile",
    "output_format",
    "json_values",
    "map_values",
    "literal_keys",
    "time_units",
    "spend_totals",
    "trace_rollups",
    "sampling",
])]
pub(super) struct QueryGuide<'a> {
    pub tables: &'a [TableSchema],
    pub normalized_fields: &'a [NormalizedFieldDefinition],
    pub metadata: &'a MetadataCatalog,
    pub attributes: &'a [AttributeCatalog],
}

#[derive(Serialize)]
pub(super) struct Example {
    name: String,
    sql: String,
}

impl QueryGuide<'_> {
    pub fn examples(&self) -> Result<[Example; 5], Error> {
        Ok([
            Example {
                name: render(&self.as_recent_spans_name())?,
                sql: render(&self.as_recent_spans_sql())?,
            },
            Example {
                name: render(&self.as_custom_metadata_name())?,
                sql: render(&self.as_custom_metadata_sql())?,
            },
            Example {
                name: render(&self.as_nested_metadata_name())?,
                sql: render(&self.as_nested_metadata_sql())?,
            },
            Example {
                name: render(&self.as_correlated_calls_name())?,
                sql: render(&self.as_correlated_calls_sql())?,
            },
            Example {
                name: render(&self.as_discover_keys_name())?,
                sql: render(&self.as_discover_keys_sql())?,
            },
        ])
    }

    pub fn gotchas(&self) -> Result<[String; 11], Error> {
        Ok([
            render(&self.as_time_window())?,
            render(&self.as_reader_limits())?,
            render(&self.as_reader_profile())?,
            render(&self.as_output_format())?,
            render(&self.as_json_values())?,
            render(&self.as_map_values())?,
            render(&self.as_literal_keys())?,
            render(&self.as_time_units())?,
            render(&self.as_spend_totals())?,
            render(&self.as_trace_rollups())?,
            render(&self.as_sampling())?,
        ])
    }
}

pub(super) fn render(template: &impl Template) -> Result<String, Error> {
    template.render().map_err(|_| Error::InvalidResponse)
}
