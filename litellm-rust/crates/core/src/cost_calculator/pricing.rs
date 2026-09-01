use std::sync::OnceLock;

use super::types::{ModelPricing, PricingDatabase};

static PRICING_DB: OnceLock<PricingDatabase> = OnceLock::new();

pub fn get_pricing_db() -> &'static PricingDatabase {
    PRICING_DB.get_or_init(|| load_pricing_from_env().unwrap_or_default())
}

fn load_pricing_from_env() -> Option<PricingDatabase> {
    let path = std::env::var("LITELLM_MODEL_PRICING_PATH").ok()?;
    let json = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&json).ok()
}

pub fn load_pricing_from_str(json: &str) -> Option<PricingDatabase> {
    serde_json::from_str(json).ok()
}

pub fn init_pricing_db(db: PricingDatabase) {
    let _ = PRICING_DB.set(db);
}

pub fn lookup_model_pricing(model: &str) -> Option<&'static ModelPricing> {
    let db = get_pricing_db();

    if let Some(pricing) = db.get(model) {
        return Some(pricing);
    }

    let stripped = strip_provider_prefix(model);
    if stripped != model
        && let Some(pricing) = db.get(stripped)
    {
        return Some(pricing);
    }

    None
}

pub(crate) fn strip_provider_prefix(model: &str) -> &str {
    let prefixes = [
        "anthropic/",
        "openai/",
        "bedrock/",
        "azure/",
        "azure_ai/",
        "gemini/",
        "vertex_ai/",
        "databricks/",
        "fireworks_ai/",
        "deepseek/",
        "groq/",
        "mistral/",
        "cohere/",
        "perplexity/",
        "xai/",
        "together_ai/",
    ];
    for prefix in &prefixes {
        if let Some(stripped) = model.strip_prefix(prefix) {
            return stripped;
        }
    }
    model
}
