use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_model_catalog::Catalog;
use litellm_router_core::{Deployment, Error, RegistryHost, host::ModelInfoLookup};
use serde_json::{Map, Value, json};

/// The reads one deployment's `get_router_model_info` makes, answered from `catalog` and the
/// deployment. Which model a wildcard's pattern routed a name to and the limits a background
/// refresh discovered are not known here, so those two answer nothing, and the group-info reads
/// are never made by this merge.
pub(super) struct NativeModelInfoHost<'a> {
    pub(super) catalog: &'a Catalog,
    pub(super) deployment: &'a Deployment,
}

fn unmapped(model: &str) -> Error {
    Error::ModelInfo {
        exception: "Exception",
        detail: format!("This model isn't mapped yet. model={model}"),
    }
}

impl NativeModelInfoHost<'_> {
    /// `get_model_info(model=...)` over the names it tries, with whether a generalization rule
    /// made the row. The deployment's own model always has a row, since Python's router registers
    /// it into the cost map when it adds the deployment, and that row holds no window of its own.
    fn row(&self, model: &str) -> Option<(Map<String, Value>, bool)> {
        let (bare, provider) = get_custom_llm_provider(model, None)
            .map_or((model, ""), |resolved| {
                (resolved.model, resolved.custom_llm_provider)
            });
        let combined = format!("{provider}/{bare}");
        let names = [combined.as_str(), model, bare];
        let registered = self.deployment.params.model.as_str();
        let Some(fields) = self.catalog.model_info(&names, provider) else {
            return names.contains(&registered).then(|| {
                (
                    Map::from_iter([("key".to_owned(), json!(registered))]),
                    false,
                )
            });
        };
        let listed = names
            .iter()
            .find_map(|name| self.catalog.lookup(name))
            .map(|found| found.canonical_key);
        let key = ("key".to_owned(), json!(listed.unwrap_or(model)));
        Some((fields.into_iter().chain([key]).collect(), listed.is_none()))
    }
}

impl RegistryHost for NativeModelInfoHost<'_> {
    async fn model_info(&self, lookup: &ModelInfoLookup) -> Result<Value, Error> {
        Ok(match lookup {
            ModelInfoLookup::ModelInfo {
                model,
                tolerate,
                exact,
            } => {
                let model = model.as_str().unwrap_or_default();
                match (self.row(model), tolerate, exact) {
                    (Some((row, generalized)), _, true) => {
                        json!({"row": row, "generalized": generalized})
                    }
                    (Some((row, _)), _, false) => Value::Object(row),
                    (None, true, _) => json!({"failed": true}),
                    (None, false, _) => return Err(unmapped(model)),
                }
            }
            ModelInfoLookup::CostEntry { key, .. } => match self.catalog.lookup(key) {
                Some(found) if found.matched_key == key => json!({"entry": found.entry.fields()}),
                _ => Value::Null,
            },
            ModelInfoLookup::Provider { model } => {
                let declared = self.deployment.params.custom_llm_provider.as_deref();
                let resolved =
                    get_custom_llm_provider(model, declared).ok_or_else(|| unmapped(model))?;
                json!([resolved.model, resolved.custom_llm_provider])
            }
            ModelInfoLookup::UserModelInfo => Value::Object(self.deployment.model_info.raw.clone()),
            ModelInfoLookup::BedrockBase { model_id } => Value::String(model_id.clone()),
            ModelInfoLookup::Log { level, message } => {
                tracing::debug!(target: "litellm::router", ?level, message, "model info");
                Value::Null
            }
            ModelInfoLookup::PatternModels { .. }
            | ModelInfoLookup::Discovered { .. }
            | ModelInfoLookup::GroupParams { .. }
            | ModelInfoLookup::GroupDeployment { .. }
            | ModelInfoLookup::GroupCapabilities { .. }
            | ModelInfoLookup::GroupSeed { .. }
            | ModelInfoLookup::ReasoningIntersect { .. } => Value::Null,
        })
    }
}
