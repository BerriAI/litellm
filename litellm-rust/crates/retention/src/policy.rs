use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    SpendLogs,
    SpendLogToolIndex,
    AutoRouterSession,
    AutoRouterUserSession,
    HealthCheck,
    DailyTagSpend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    pub target: Target,
    pub max_age: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub batch_size: u32,
    pub max_batches: u32,
    pub run_budget: Duration,
    pub batch_timeout: Duration,
    pub batch_pause: Duration,
    pub max_consecutive_failures: u32,
    pub failure_backoff: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionInterval {
    Day,
    Week,
    Month,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartitionPlan {
    pub interval: PartitionInterval,
    pub periods_ahead: u32,
}
