/// The registry document this build serves when no other one is loaded.
pub const BUNDLED_REGISTRY: &[u8] =
    include_bytes!("../../../../../model_prices_and_context_window.json");

/// How many models `BUNDLED_REGISTRY` holds, counted when the crate is built so a fetched document can be size
/// checked without reading the bundled one
pub const BUNDLED_MODEL_COUNT: usize =
    include!(concat!(env!("OUT_DIR"), "/bundled_model_count.rs"));
