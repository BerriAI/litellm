//! The deployment build: how configured entries become the registry's deployments.
//!
//! The core owns the loop (`set_model_list`, `_create_deployment` and `_add_deployment`): which
//! values are secret references, the id of a deployment that declares none, one deployment per
//! `organization`, and the order everything happens in. What needs the embedding process is a
//! host step, asked for one deployment at a time so an error surfaces exactly where the Python
//! router raises it:
//! - [`Host::secrets`]: the `os.environ/` references the entry holds;
//! - [`Host::validate`]: the host's checks, and the entry as it normalizes it;
//! - [`Host::provider`]: the provider that serves it;
//! - [`Host::added`]: the host's side effects, and the entry as the registry stores it.

use serde_json::{Map, Value};

use crate::{
    error::{ConfigError, Error},
    host::{BuildAsk, BuildPath, Built, Host, Provided, SecretAt, SecretRef},
    registry::{DeploymentSpec, deployment_id},
};

/// The `litellm_params` keys whose `os.environ/` value is a secret pointer the identity source
/// dereferences at use time, so the build leaves it as written. The router's copy of
/// `_WIF_POINTER_FIELDS` (litellm/types/router.py), which the proxy's config loader also reads.
pub const SECRET_POINTER_PARAMS: &[&str] = &[
    "anthropic_issuer_signing_key_ref",
    "anthropic_keycloak_client_secret_ref",
];

const SECRET_PREFIX: &str = "os.environ/";

/// How a build's deployments join the registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum Install {
    /// They become every deployment (`set_model_list`).
    Replace,
    /// They are added at the end (a deployment retried once its provider resolves).
    Append,
    /// Each replaces the deployment with its id, if any, and is added at the end
    /// (`add_deployment`, an upsert).
    Upsert,
}

/// One deployment the build accepted, as the registry stores it.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltDeployment {
    /// The entry index in the list the build was given.
    pub index: usize,
    pub spec: DeploymentSpec,
}

/// Build `entries` (model_list entries) through `host`, in order. `resolve_secrets` resolves
/// top-level `os.environ/` references first, as `set_model_list` does; `add_deployment` and
/// upserts take their params as given. `install` takes each deployment as soon as it is
/// accepted, so it joins the registry before the next one is built, as the Python router adds
/// each in turn.
pub async fn build(
    host: &dyn Host,
    path: BuildPath,
    entries: Vec<Map<String, Value>>,
    resolve_secrets: bool,
    mut install: impl FnMut(&BuiltDeployment) -> Result<(), Error> + Send,
) -> Result<Vec<BuiltDeployment>, Error> {
    let declared: Vec<Option<String>> = entries.iter().map(declared_id).collect();
    let mut built = Vec::new();
    for (index, mut entry) in entries.into_iter().enumerate() {
        let model_name = text(entry.get("model_name"));
        let mut params = match entry.remove("litellm_params") {
            Some(Value::Object(params)) => params,
            _ => Map::new(),
        };
        if resolve_secrets {
            let refs = param_refs(&params);
            if !refs.is_empty() {
                let values = host.secrets(index, &refs).await?;
                for (secret, value) in refs.into_iter().zip(values) {
                    if let SecretAt::Param(key) = secret.at {
                        params.insert(key, value);
                    }
                }
            }
        }
        let mut info = match entry.remove("model_info") {
            Some(Value::Object(info)) => info,
            _ => Map::new(),
        };
        let declared_id = declared[index].clone();
        if !info.contains_key("id") {
            info.insert(
                "id".to_owned(),
                Value::String(deployment_id(&model_name, &params)),
            );
        }
        // `set_model_list` adds one deployment per organization of a list, each with the
        // organization set on the same params (so every one keeps the list's id).
        let organizations = match (path, params.get("organization")) {
            (BuildPath::Create, Some(Value::Array(organizations))) => organizations.clone(),
            _ => vec![Value::Null],
        };
        let duplicate_id = path == BuildPath::Create
            && declared_id.as_ref().is_some_and(|id| {
                declared
                    .iter()
                    .filter(|other| other.as_ref() == Some(id))
                    .count()
                    > 1
            });
        for organization in organizations {
            if !organization.is_null() {
                params.insert("organization".to_owned(), organization);
            }
            let mut deployment = entry.clone();
            deployment.insert("model_name".to_owned(), Value::String(model_name.clone()));
            deployment.insert("litellm_params".to_owned(), Value::Object(params.clone()));
            deployment.insert("model_info".to_owned(), Value::Object(info.clone()));
            if let Some(spec) = build_one(
                host,
                path,
                index,
                &deployment,
                declared_id.as_deref(),
                duplicate_id,
            )
            .await?
            {
                let deployment = BuiltDeployment { index, spec };
                install(&deployment)?;
                built.push(deployment);
            }
        }
    }
    Ok(built)
}

/// One deployment's steps: validate, provider, the data-source references, added.
async fn build_one(
    host: &dyn Host,
    path: BuildPath,
    index: usize,
    entry: &Map<String, Value>,
    declared_id: Option<&str>,
    duplicate_id: bool,
) -> Result<Option<DeploymentSpec>, Error> {
    let ask = |entry| BuildAsk {
        path,
        index,
        entry,
        declared_id,
        duplicate_id,
    };
    let validated = match host.validate(&ask(entry)).await? {
        Built::Entry(validated) => validated,
        Built::Skip | Built::Invalid => return Ok(None),
    };
    let provider = match host.provider(&ask(&validated)).await? {
        Provided::Provider(provider) => provider,
        Provided::Invalid => return Ok(None),
    };
    let refs = data_source_refs(&validated);
    if !refs.is_empty() {
        let _ = host.secrets(index, &refs).await?;
    }
    let stored = match host.added(&ask(&validated), provider.as_deref()).await? {
        Built::Entry(stored) => stored,
        Built::Skip | Built::Invalid => return Ok(None),
    };
    let model_name = text(stored.get("model_name"));
    let mut spec: DeploymentSpec = serde_json::from_value(Value::Object(stored))
        .map_err(|source| ConfigError::InvalidDeployment { model_name, source })?;
    spec.provider = provider;
    Ok(Some(spec))
}

/// The top-level `litellm_params` values `set_model_list` resolves.
fn param_refs(params: &Map<String, Value>) -> Vec<SecretRef> {
    params
        .iter()
        .filter(|(key, _)| !SECRET_POINTER_PARAMS.contains(&key.as_str()))
        .filter_map(|(key, value)| {
            let name = value
                .as_str()
                .filter(|name| name.starts_with(SECRET_PREFIX))?;
            Some(SecretRef {
                name: name.to_owned(),
                at: SecretAt::Param(key.clone()),
            })
        })
        .collect()
}

/// The `dataSources[].parameters.endpoint` / `key` references `_add_deployment` reads from the
/// environment.
fn data_source_refs(entry: &Map<String, Value>) -> Vec<SecretRef> {
    let sources = entry
        .get("litellm_params")
        .and_then(|params| params.get("dataSources"))
        .and_then(Value::as_array);
    let mut refs = Vec::new();
    for (source, data_source) in sources.into_iter().flatten().enumerate() {
        let Some(parameters) = data_source.get("parameters").and_then(Value::as_object) else {
            continue;
        };
        for key in ["endpoint", "key"] {
            if let Some(name) = parameters
                .get(key)
                .and_then(Value::as_str)
                .filter(|name| name.starts_with(SECRET_PREFIX))
            {
                refs.push(SecretRef {
                    name: name.to_owned(),
                    at: SecretAt::DataSource {
                        source,
                        key: key.to_owned(),
                    },
                });
            }
        }
    }
    refs
}

/// The `model_info.id` an entry declares, as Python's `str()` spells it.
fn declared_id(entry: &Map<String, Value>) -> Option<String> {
    match entry.get("model_info")?.get("id")? {
        Value::Null => None,
        Value::String(id) => Some(id.clone()),
        other => Some(other.to_string()),
    }
}

/// A name as Python's `str()` spells it.
fn text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Null) | None => "None".to_owned(),
        Some(other) => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;
    use crate::{
        cache::{InMemoryRouterCache, RouterCache},
        error::AttemptError,
        host::{Attempt, Output, RouterEvent, RouterRequest, RoutingContext, SelectStage},
        registry::Deployment,
    };

    /// Records every build step and resolves `os.environ/X` to `secret-X`.
    #[derive(Default)]
    struct Recording {
        steps: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Host for Recording {
        async fn invoke(&self, _attempt: &Attempt) -> Result<Output, AttemptError> {
            unreachable!("the build runs no attempt")
        }
        async fn select(
            &self,
            _stage: &SelectStage,
            _ctx: &RoutingContext<'_>,
            candidates: Vec<Arc<Deployment>>,
        ) -> Result<Vec<Arc<Deployment>>, Error> {
            Ok(candidates)
        }
        async fn allow_fallback(
            &self,
            _request: &RouterRequest,
            _target: &str,
        ) -> Result<bool, crate::error::AttemptError> {
            Ok(true)
        }
        async fn sleep(&self, _duration: std::time::Duration) {}
        fn observe(&self, _event: &RouterEvent) {}
        fn cache(&self) -> Arc<dyn RouterCache> {
            Arc::new(InMemoryRouterCache::default())
        }
        async fn secrets(&self, _index: usize, refs: &[SecretRef]) -> Result<Vec<Value>, Error> {
            self.steps.lock().unwrap().push(format!("secrets {refs:?}"));
            Ok(refs
                .iter()
                .map(|r| json!(format!("secret-{}", &r.name[SECRET_PREFIX.len()..])))
                .collect())
        }
        async fn validate(&self, ask: &BuildAsk<'_>) -> Result<Built, Error> {
            self.steps.lock().unwrap().push(format!(
                "validate {} {}",
                ask.index, ask.entry["litellm_params"]
            ));
            Ok(Built::Entry(ask.entry.clone()))
        }
        async fn provider(&self, ask: &BuildAsk<'_>) -> Result<Provided, Error> {
            self.steps
                .lock()
                .unwrap()
                .push(format!("provider {}", ask.index));
            Ok(Provided::Provider(Some("azure".to_owned())))
        }
        async fn added(&self, ask: &BuildAsk<'_>, provider: Option<&str>) -> Result<Built, Error> {
            self.steps
                .lock()
                .unwrap()
                .push(format!("added {} {provider:?}", ask.index));
            Ok(Built::Entry(ask.entry.clone()))
        }
    }

    fn entries(value: Value) -> Vec<Map<String, Value>> {
        serde_json::from_value(value).unwrap()
    }

    #[tokio::test]
    async fn create_resolves_secrets_then_builds_one_deployment_per_organization() {
        let host = Recording::default();
        let built = build(
            &host,
            BuildPath::Create,
            entries(json!([{
                "model_name": "g",
                "litellm_params": {
                    "model": "azure/x",
                    "api_key": "os.environ/KEY",
                    "anthropic_issuer_signing_key_ref": "os.environ/POINTER",
                    "organization": ["a", "b"],
                },
            }])),
            true,
            |_| Ok(()),
        )
        .await
        .unwrap();
        let steps = host.steps.lock().unwrap().clone();
        assert_eq!(steps.len(), 7, "{steps:?}");
        assert!(steps[0].starts_with("secrets") && steps[0].contains("KEY"));
        assert!(!steps[0].contains("POINTER"));
        assert!(steps[1].contains(r#""organization":"a""#));
        assert!(steps[1].contains("secret-KEY"));
        assert_eq!(steps[3], "added 0 Some(\"azure\")");
        assert!(steps[4].contains(r#""organization":"b""#));
        assert_eq!(built.len(), 2);
        assert_eq!(built[0].spec.provider.as_deref(), Some("azure"));
        assert_eq!(
            built[0].spec.model_info["id"],
            built[1].spec.model_info["id"]
        );
        let deployment = Deployment::from_spec(built[0].spec.clone()).unwrap();
        assert_eq!(deployment.provider(), "azure");
    }

    #[tokio::test]
    async fn add_leaves_params_as_given_and_asks_for_data_source_references() {
        let host = Recording::default();
        let built = build(
            &host,
            BuildPath::Add,
            entries(json!([{
                "model_name": "g",
                "litellm_params": {
                    "model": "azure/x",
                    "api_key": "os.environ/KEY",
                    "organization": ["a", "b"],
                    "dataSources": [{"parameters": {"endpoint": "os.environ/END", "key": "plain"}}],
                },
                "model_info": {"id": "given"},
            }])),
            false,
            |_| Ok(()),
        )
        .await
        .unwrap();
        let steps = host.steps.lock().unwrap().clone();
        assert_eq!(steps.len(), 4, "{steps:?}");
        assert!(steps[0].contains("os.environ/KEY"));
        assert!(steps[2].contains("DataSource { source: 0, key: \"endpoint\" }"));
        assert_eq!(built.len(), 1);
        assert_eq!(built[0].spec.model_info["id"], "given");
    }
}
