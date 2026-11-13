mod auth;
mod error;
mod includes;
mod mcp;
mod model;
mod redis;
mod settings;
mod value;

use std::path::Path;

use serde::Deserialize;

pub use auth::{
    ClaimPath, JwtAuthSettings, JwtIssuerConfig, JwtLitellmRoleMap, JwtRoutingOverride,
    LitellmRole, OneOrMany, RbacRole, RoleMapping, RolePermissions, RoutingPath, ScopeMapping,
    UnregisteredJwtClientBehavior,
};
pub use error::Error;
pub use litellm_router_types::{LitellmParams, Spelled};
pub use mcp::{McpAuth, McpServer, McpTransport};
pub use model::Model;
pub use redis::{CacheSettings, RedisSettings};
pub use settings::{
    ClickHouseStoreSettings, GeneralSettings, LenientFlag, LiteLlmSettings, RouterSettings,
    TracingSettings, TracingStoreSettings, UserHeaderMapping,
};
pub use value::{AdditionalFields, Object, Value};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model_list: Box<[Model]>,
    pub general_settings: GeneralSettings,
    pub router_settings: RouterSettings,
    pub litellm_settings: LiteLlmSettings,
    pub environment_variables: Object,
    pub callback_settings: Object,
    pub assistant_settings: Object,
    pub default_vertex_config: Object,
    pub mcp_servers: std::collections::BTreeMap<String, McpServer>,
    pub credential_list: Box<[Object]>,
    pub guardrails: Box<[Object]>,
    pub prompts: Box<[Object]>,
    pub sandbox_tools: Box<[Object]>,
    pub search_tools: Box<[Object]>,
    pub files_settings: Box<[Object]>,
    pub finetune_settings: Box<[Object]>,
    pub mcp_tools: Box<[Object]>,
    pub vector_store_registry: Box<[Object]>,
    pub worker_registry: Box<[Object]>,
    pub agents: Box<[Object]>,
    pub agent_list: Box<[Object]>,
    pub policies: Object,
    pub policy_attachments: Box<[Object]>,
    pub include: Box<[String]>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

impl Config {
    pub fn from_yaml(yaml: &str) -> Result<Self, Error> {
        Ok(serde_yaml_ng::from_str(yaml)?)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(serde_yaml_ng::from_value(includes::load(path.as_ref())?)?)
    }
}
