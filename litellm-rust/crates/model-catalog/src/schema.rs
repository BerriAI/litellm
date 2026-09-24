use crate::model_info::ModelInfo;

/// JSON Schema derived from the typed model entry for consumers that need one.
pub fn model_entry_json_schema() -> schemars::Schema {
    schemars::schema_for!(ModelInfo)
}
