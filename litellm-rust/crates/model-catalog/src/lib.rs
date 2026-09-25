mod catalog;
mod error;
mod model_info;
#[cfg(feature = "schema")]
mod schema;

pub use catalog::{AliasIssue, Catalog, IntegrityLimits, ModelEntry, ModelMatch, Provenance};
pub use error::Error;
pub use model_info::{
    AudioFormat, FallbackGeneralizations, FallbackRule, InputModality, Mode, ModelInfo,
    OffPeakPricing, OffPeakWindow, OutputModality, ReasoningEffort, SearchContextCostPerQuery,
    TieredRate, UtcHours, VertexAiAudioApi, WebSearchBillingUnit, Weekday,
};

#[cfg(feature = "schema")]
pub use schema::model_entry_json_schema;
