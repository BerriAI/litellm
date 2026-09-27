use time::{Date, OffsetDateTime};

pub trait Key: Ord + Clone + Send + Sync + 'static {}

impl<T: Ord + Clone + Send + Sync + 'static> Key for T {}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntityKey {
    User(String),
    EndUser(String),
    Key(String),
    Team(String),
    TeamMember {
        team_id: String,
        user_id: String,
    },
    Organization(String),
    OrganizationMember {
        organization_id: String,
        user_id: String,
    },
    Project(String),
    Tag(String),
    ModelAccessGroup(String),
    Agent(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DailyEntity {
    User(String),
    Team(String),
    Organization(String),
    EndUser(String),
    Agent(String),
    Tag(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DailyKey {
    pub date: Date,
    pub entity: DailyEntity,
    pub api_key: String,
    pub model: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub mcp_namespaced_tool_name: Option<String>,
    pub endpoint: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WindowKey {
    pub entity: EntityKey,
    pub duration: String,
    pub start: OffsetDateTime,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CounterKey {
    pub entity: EntityKey,
    pub window: Option<String>,
}

impl CounterKey {
    pub fn lifetime(entity: EntityKey) -> Self {
        Self {
            entity,
            window: None,
        }
    }
}
