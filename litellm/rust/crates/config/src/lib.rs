mod auth;
mod environment;
mod error;
mod layer;
mod mcp;
mod model;
mod pass_through;
mod priority;
mod redis;
mod routes;
mod settings;
mod value;

use serde::Deserialize;

pub use auth::{
    ClaimPath, JwtAuthSettings, JwtIssuerConfig, JwtLitellmRoleMap, JwtRoutingOverride,
    LitellmRole, OneOrMany, RbacRole, RoleMapping, RolePermissions, RoutingPath, ScopeMapping,
    UnregisteredJwtClientBehavior,
};
pub use environment::{Environment, Flag};
pub use error::Error;
pub use layer::Origins;
pub use litellm_router_types::{
    LitellmParams, RouterRedis, RouterSettings, RoutingGroupSpec, RoutingStrategy, Spelled,
};
pub use mcp::{McpAuth, McpServer, McpTransport};
pub use model::Model;
pub use pass_through::{PassThroughEndpoint, PassThroughGuardrail};
pub use priority::{
    BorrowingEntity, ClassShareEntity, ModelLimitBorrowingConfig, PriorityClassShareConfig,
    PriorityReservation, PriorityReservationSettings, ReservationUnit,
};
pub use redis::{CacheSettings, RedisSettings};
pub use routes::RouteList;
pub use settings::{
    ClickHouseStoreSettings, GeneralSettings, LiteLlmSettings, TracingSettings,
    TracingStoreSettings, UserHeaderMapping,
};
pub use value::{AdditionalFields, Object, Value};

pub const ENV_REFERENCE_PREFIX: &str = "os.environ/";

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
    /// Which keys the environment set, once `with_environment` has put it under this config.
    #[serde(skip)]
    pub origins: Origins,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

impl Config {
    pub fn from_yaml(yaml: &str) -> Result<Self, Error> {
        Self::from_value(litellm_yaml::from_str(yaml)?)
    }

    pub fn from_value(document: Value) -> Result<Self, Error> {
        Ok(serde_yaml_ng::from_value(document)?)
    }
}
