#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    InProgress,
    Cancelling,
    Completed,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq)]
pub struct BatchRequestCounts {
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Eq)]
pub struct BatchResponse {
    pub id: String,
    pub object: String,
    pub endpoint: String,
    pub input_file_id: String,
    pub completion_window: String,
    pub status: BatchStatus,
    pub output_file_id: String,
    pub created_at: i64,
    pub in_progress_at: Option<i64>,
    pub expires_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub expired_at: Option<i64>,
    pub cancelling_at: Option<i64>,
    pub cancelled_at: Option<i64>,
    pub request_counts: BatchRequestCounts,
}
