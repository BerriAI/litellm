use std::{collections::VecDeque, sync::Mutex};

use time::OffsetDateTime;

use crate::Outcome;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SerializedJson(pub String);

#[derive(Clone, Debug, PartialEq)]
pub struct SpendLogRow {
    pub request_id: String,
    pub call_type: String,
    pub api_key: String,
    pub spend: f64,
    pub total_tokens: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub start_time: OffsetDateTime,
    pub end_time: OffsetDateTime,
    pub completion_start_time: Option<OffsetDateTime>,
    pub request_duration_ms: Option<u64>,
    pub model: String,
    pub model_id: Option<String>,
    pub model_group: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub api_base: Option<String>,
    pub user: Option<String>,
    pub team_id: Option<String>,
    pub organization_id: Option<String>,
    pub end_user: Option<String>,
    pub agent_id: Option<String>,
    pub requester_ip_address: Option<String>,
    pub session_id: Option<String>,
    pub litellm_call_id: Option<String>,
    pub mcp_namespaced_tool_name: Option<String>,
    pub cache_hit: Option<bool>,
    pub cache_key: Option<String>,
    pub request_tags: Vec<String>,
    pub outcome: Outcome,
    pub metadata: SerializedJson,
    pub messages: SerializedJson,
    pub response: SerializedJson,
    pub proxy_server_request: SerializedJson,
}

impl SpendLogRow {
    pub fn body_bytes(&self) -> usize {
        self.request_id.len()
            + self.metadata.0.len()
            + self.messages.0.len()
            + self.response.0.len()
            + self.proxy_server_request.0.len()
    }
}

#[derive(Debug, Default)]
struct Queued {
    rows: VecDeque<SpendLogRow>,
    bytes: usize,
}

impl Queued {
    fn evict_oldest_over(&mut self, max_bytes: usize) -> usize {
        let before = self.rows.len();
        while self.bytes > max_bytes && self.rows.len() > 1 {
            let Some(oldest) = self.rows.pop_front() else {
                break;
            };
            self.bytes -= oldest.body_bytes();
        }
        before - self.rows.len()
    }
}

#[derive(Debug)]
pub struct LogQueue {
    max_bytes: usize,
    queued: Mutex<Queued>,
}

impl LogQueue {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            queued: Mutex::default(),
        }
    }

    fn with_queued<T>(&self, f: impl FnOnce(&mut Queued) -> T) -> T {
        f(&mut self
            .queued
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()))
    }

    pub fn enqueue(&self, row: SpendLogRow) -> usize {
        self.with_queued(|queued| {
            queued.bytes += row.body_bytes();
            queued.rows.push_back(row);
            queued.evict_oldest_over(self.max_bytes)
        })
    }

    pub fn requeue(&self, rows: Vec<SpendLogRow>) -> usize {
        self.with_queued(|queued| {
            for row in rows.into_iter().rev() {
                queued.bytes += row.body_bytes();
                queued.rows.push_front(row);
            }
            queued.evict_oldest_over(self.max_bytes)
        })
    }

    pub fn dequeue(&self, max_rows: usize) -> Vec<SpendLogRow> {
        self.with_queued(|queued| {
            let take = max_rows.min(queued.rows.len());
            let rows: Vec<_> = queued.rows.drain(..take).collect();
            queued.bytes -= rows.iter().map(SpendLogRow::body_bytes).sum::<usize>();
            rows
        })
    }

    pub fn len(&self) -> usize {
        self.with_queued(|queued| queued.rows.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
