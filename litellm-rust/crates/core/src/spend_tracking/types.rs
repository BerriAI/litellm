use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A per-request spend log entry. Maps to `LiteLLM_SpendLogs` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendEntry {
    pub request_id: String,
    pub call_type: String,
    pub api_key: String,
    pub spend: f64,
    pub total_tokens: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub model: String,
    pub user: Option<String>,
    pub team_id: Option<String>,
    pub organization_id: Option<String>,
    pub end_user: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub status: SpendStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpendStatus {
    Success,
    Failure,
}

/// An incremental spend update for a specific entity.
/// Used to batch-update spend counters in the DB/Redis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendUpdateItem {
    pub entity_type: EntityType,
    pub entity_id: String,
    pub cost: f64,
}

/// Entity types that track spend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Key,
    User,
    EndUser,
    Team,
    TeamMember,
    Organization,
    Tag,
    Agent,
}

/// Aggregated spend updates grouped by entity type.
/// Produced by the background worker after batching.
#[derive(Debug, Clone, Default)]
pub struct SpendUpdateBatch {
    pub key_updates: HashMap<String, f64>,
    pub user_updates: HashMap<String, f64>,
    pub end_user_updates: HashMap<String, f64>,
    pub team_updates: HashMap<String, f64>,
    pub team_member_updates: HashMap<String, f64>,
    pub org_updates: HashMap<String, f64>,
    pub tag_updates: HashMap<String, f64>,
    pub agent_updates: HashMap<String, f64>,
    pub spend_logs: Vec<SpendEntry>,
}

impl SpendUpdateBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_update(&mut self, item: SpendUpdateItem) {
        let map = match item.entity_type {
            EntityType::Key => &mut self.key_updates,
            EntityType::User => &mut self.user_updates,
            EntityType::EndUser => &mut self.end_user_updates,
            EntityType::Team => &mut self.team_updates,
            EntityType::TeamMember => &mut self.team_member_updates,
            EntityType::Organization => &mut self.org_updates,
            EntityType::Tag => &mut self.tag_updates,
            EntityType::Agent => &mut self.agent_updates,
        };
        *map.entry(item.entity_id).or_insert(0.0) += item.cost;
    }

    pub fn add_spend_log(&mut self, entry: SpendEntry) {
        self.spend_logs.push(entry);
    }

    pub fn is_empty(&self) -> bool {
        self.key_updates.is_empty()
            && self.user_updates.is_empty()
            && self.end_user_updates.is_empty()
            && self.team_updates.is_empty()
            && self.team_member_updates.is_empty()
            && self.org_updates.is_empty()
            && self.tag_updates.is_empty()
            && self.agent_updates.is_empty()
            && self.spend_logs.is_empty()
    }

    pub fn total_entries(&self) -> usize {
        self.key_updates.len()
            + self.user_updates.len()
            + self.end_user_updates.len()
            + self.team_updates.len()
            + self.team_member_updates.len()
            + self.org_updates.len()
            + self.tag_updates.len()
            + self.agent_updates.len()
            + self.spend_logs.len()
    }
}
