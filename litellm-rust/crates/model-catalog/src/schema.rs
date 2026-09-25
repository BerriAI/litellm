use crate::model_info::ModelInfo;

/// JSON Schema for one catalog model entry, mirroring
/// `model_prices_and_context_window.schema.json`'s `modelEntry` definition.
pub fn model_entry_json_schema() -> schemars::Schema {
    schemars::schema_for!(ModelInfo)
}
