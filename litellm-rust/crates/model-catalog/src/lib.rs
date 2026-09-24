mod capabilities;
mod catalog;
mod error;
mod fallback;
mod model_info;
mod pricing;
#[cfg(feature = "schema")]
mod schema;

pub use capabilities::{
    AudioFormat, InputModality, Mode, OutputModality, ReasoningEffort, VertexAiAudioApi,
};
pub use catalog::{AliasIssue, Catalog, IntegrityLimits, ModelEntry, ModelMatch, Provenance};
pub use error::Error;
pub use fallback::{FallbackGeneralizations, FallbackRule};
pub use model_info::ModelInfo;
pub use pricing::{
    OffPeakPricing, OffPeakWindow, SearchContextCostPerQuery, TieredRate, UtcHours,
    WebSearchBillingUnit, Weekday,
};

#[cfg(feature = "schema")]
pub use schema::model_entry_json_schema;
