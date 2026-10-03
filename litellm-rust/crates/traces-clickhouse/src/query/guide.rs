use askama::Template;
use litellm_traces::query::guide::Example;

use super::{AttributeCatalog, Discovery, MetadataCatalog, TableSchema};
use crate::{Error, NormalizedFieldDefinition, query_access::ReaderLimits};

#[derive(Template)]
#[template(path = "query_help.jinja", escape = "none", blocks = [
    "live_schema",
    "normalized_fields",
    "metadata",
    "attributes",
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
    "recent_spend_name",
    "recent_spend_sql",
    "model_spend_name",
    "model_spend_sql",
    "trace_spend_name",
    "trace_spend_sql",
    "unmatched_spans_name",
    "unmatched_spans_sql",
    "trace_summary_name",
    "trace_summary_sql",
    "failed_spans_name",
    "failed_spans_sql",
    "metadata_filter_name",
    "metadata_filter_sql",
    "missing_spend",
    "partial_spend",
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
    pub limits: &'a ReaderLimits,
}

impl QueryGuide<'_> {
    pub fn sections(&self) -> Result<[String; 4], Error> {
        Ok([
            render(&self.as_live_schema())?,
            render(&self.as_normalized_fields())?,
            render(&self.as_metadata())?,
            render(&self.as_attributes())?,
        ])
    }

    pub fn examples(&self) -> Result<[Example; 12], Error> {
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
            Example {
                name: render(&self.as_recent_spend_name())?,
                sql: render(&self.as_recent_spend_sql())?,
            },
            Example {
                name: render(&self.as_model_spend_name())?,
                sql: render(&self.as_model_spend_sql())?,
            },
            Example {
                name: render(&self.as_trace_spend_name())?,
                sql: render(&self.as_trace_spend_sql())?,
            },
            Example {
                name: render(&self.as_unmatched_spans_name())?,
                sql: render(&self.as_unmatched_spans_sql())?,
            },
            Example {
                name: render(&self.as_trace_summary_name())?,
                sql: render(&self.as_trace_summary_sql())?,
            },
            Example {
                name: render(&self.as_failed_spans_name())?,
                sql: render(&self.as_failed_spans_sql())?,
            },
            Example {
                name: render(&self.as_metadata_filter_name())?,
                sql: render(&self.as_metadata_filter_sql())?,
            },
        ])
    }

    pub fn gotchas(&self) -> Result<[String; 13], Error> {
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
            render(&self.as_missing_spend())?,
            render(&self.as_partial_spend())?,
            render(&self.as_trace_rollups())?,
            render(&self.as_sampling())?,
        ])
    }
}

pub(super) fn render(template: &impl Template) -> Result<String, Error> {
    template.render().map_err(|_| Error::InvalidResponse)
}
