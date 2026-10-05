macro_rules_attribute::attribute_alias! {
    #[apply(wire_type)] =
        #[derive(serde::Serialize, serde::Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))];
    #[apply(response_type)] =
        #[derive(serde::Serialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))];
    #[apply(request_type)] =
        #[derive(serde::Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))];
}

mod error;
mod normalize;
mod otlp;
pub mod query;
mod query_access;
mod resolve;
#[cfg(feature = "schema")]
pub mod schema;
mod shared;
mod tenant;
mod truncate;
mod ui;
mod view;
pub mod wire;

pub use error::{Error, InvalidCallKey, InvalidQuery, InvalidScope};
pub use normalize::{
    AgentMetadata, AgentType, CallEvidence, CallEvidenceKind, CallKey, Integration, NormalizedSpan,
    ObservationType,
};
pub use otlp::{DecodeLimits, DecodedEvent, DecodedSpan, decode_otlp, decode_otlp_with_limits};
pub use query::ReadQuery;
pub use query_access::QueryScope;
pub use resolve::{SpendLookup, iso_time, listed_summary, resolve_trace};
pub use shared::{Shared, SharedIdentity};
pub use tenant::Tenant;
pub use truncate::{truncate_messages, truncate_value};
pub use ui::{ChatRole, UiContent, UiField, UiMessage, UiToolCall, to_ui_content};
pub use view::{
    AgentNode, Span, SpanDetail, SpanErrorPage, SpanStatus, Trace, TracePage, TraceSummary,
};
