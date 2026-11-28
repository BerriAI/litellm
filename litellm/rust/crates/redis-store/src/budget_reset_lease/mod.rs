//! The coordination lease shared with Python's ResetBudgetJob.

use std::{sync::LazyLock, time::Duration};

use litellm_python_compat::{Value, json};
use litellm_redis_client::{Error, RedisClient, Script, commands};
use tokio::time::Instant;
use uuid::Uuid;

const KEY: &str = "cronjob_lock:reset_budget_job";
const TTL: Duration = Duration::from_secs(900);
const SWEEP_TIME: Duration = Duration::from_secs(870);
static RELEASE: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"#,
    )
});

/// An acknowledged lease. A failed or ambiguous acquisition never yields ownership.
pub struct Lease<'a> {
    redis: &'a RedisClient,
    owner: String,
    deadline: Instant,
}

impl<'a> Lease<'a> {
    pub async fn acquire(redis: &'a RedisClient) -> Result<Option<Self>, Error> {
        let owner =
            json::dumps(&Value::Str(Uuid::new_v4().to_string())).expect("a UUID is JSON-encodable");
        let deadline = Instant::now() + SWEEP_TIME;
        let accepted = redis.send(commands::set_nx(KEY, &owner, TTL)).await?;
        Ok(accepted.then_some(Self {
            redis,
            owner,
            deadline,
        }))
    }

    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    pub async fn is_current(&self) -> Result<bool, Error> {
        if Instant::now() >= self.deadline {
            return Ok(false);
        }
        let current = self.redis.send(commands::get::<String>(KEY)).await?;
        Ok(current.as_deref() == Some(self.owner.as_str()) && Instant::now() < self.deadline)
    }

    pub async fn release(self) -> Result<(), Error> {
        self.redis
            .send(commands::eval::<i64>(&RELEASE, &[KEY], &self.owner))
            .await?;
        Ok(())
    }
}
