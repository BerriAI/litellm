use std::{sync::LazyLock, time::Duration};

use litellm_redis_client::{
    Script, Value,
    commands::{self, unexpected_reply},
};

use super::{Local, RateLimits};
use crate::{Error, Pending, Send, memory::UpdateGuard, request::pending};

const RESERVE_ENQUEUED_TOKENS: &str = r#"
local amount = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current + amount > limit then
    return {0, current}
end
local updated = redis.call('INCRBY', KEYS[1], amount)
redis.call('EXPIRE', KEYS[1], ttl)
return {1, updated}
"#;

const REFUND_ENQUEUED_TOKENS: &str = r#"
local updated = redis.call('DECRBY', KEYS[1], tonumber(ARGV[1]))
if updated <= 0 then
    redis.call('DEL', KEYS[1])
end
return 1
"#;

const SAVE_RESERVATION: &str = r#"
redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[2]))
return 1
"#;

const POP_RESERVATION: &str = r#"
local value = redis.call('GET', KEYS[1])
if value and value ~= '' then
    redis.call('SET', KEYS[1], '', 'EX', tonumber(ARGV[1]))
end
return value
"#;

static RESERVE: LazyLock<Script> = LazyLock::new(|| Script::new(RESERVE_ENQUEUED_TOKENS));
static REFUND: LazyLock<Script> = LazyLock::new(|| Script::new(REFUND_ENQUEUED_TOKENS));
static SAVE: LazyLock<Script> = LazyLock::new(|| Script::new(SAVE_RESERVATION));
static POP: LazyLock<Script> = LazyLock::new(|| Script::new(POP_RESERVATION));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnqueuedReply {
    pub reserved: bool,
    pub count: i64,
}

impl RateLimits {
    pub fn enqueued_reserve(
        &self,
        key: &str,
        amount: i64,
        ttl_seconds: i64,
        limit: i64,
    ) -> Option<Pending<EnqueuedReply>> {
        let args = vec![
            amount.to_string(),
            ttl_seconds.to_string(),
            limit.to_string(),
        ];
        Some(pending(self.batch(Send::Now)?.add(
            commands::eval::<Value>(&RESERVE, &[key], args).map(enqueued_reply),
        )))
    }

    pub fn enqueued_refund(&self, key: &str, amount: i64) -> Option<Pending<()>> {
        Some(pending(self.batch(Send::Now)?.add(
            commands::eval::<Value>(&REFUND, &[key], vec![amount.to_string()]).map(ignored),
        )))
    }

    pub fn enqueued_record_save(
        &self,
        key: &str,
        value: &str,
        ttl_seconds: i64,
    ) -> Option<Pending<()>> {
        let args = vec![value.to_owned(), ttl_seconds.to_string()];
        Some(pending(self.batch(Send::Now)?.add(
            commands::eval::<Value>(&SAVE, &[key], args).map(ignored),
        )))
    }

    pub fn enqueued_record_pop(
        &self,
        key: &str,
        ttl_seconds: i64,
    ) -> Option<Pending<Option<String>>> {
        Some(pending(self.batch(Send::Now)?.add(
            commands::eval::<Value>(&POP, &[key], vec![ttl_seconds.to_string()]).map(record),
        )))
    }

    pub fn enqueued_guard(&self) -> UpdateGuard<'_> {
        self.cache.memory().guard()
    }

    pub fn get_raw_local(&self, key: &str) -> Result<Option<Local>, Error> {
        Ok(self.cache.memory().get(key)?)
    }

    pub fn set_raw_local(&self, key: &str, value: Local, ttl: Duration) -> Result<(), Error> {
        Ok(self.cache.memory().set(key, value, Some(ttl))?)
    }

    pub fn delete_raw_local(&self, key: &str) -> Result<(), Error> {
        Ok(self.cache.memory().delete(key)?)
    }
}

fn ignored(_: Value) -> Result<(), litellm_redis_client::Error> {
    Ok(())
}

fn enqueued_reply(reply: Value) -> Result<EnqueuedReply, litellm_redis_client::Error> {
    match reply {
        Value::Array(items) => match items.as_slice() {
            [Value::Int(reserved), Value::Int(count)] => Ok(EnqueuedReply {
                reserved: *reserved == 1,
                count: *count,
            }),
            _ => Err(unexpected_reply(
                "enqueued-token script reply is not an (applied, count) pair",
            )),
        },
        _ => Err(unexpected_reply(
            "enqueued-token script reply is not a list",
        )),
    }
}

fn record(reply: Value) -> Result<Option<String>, litellm_redis_client::Error> {
    match reply {
        Value::Nil => Ok(None),
        Value::BulkString(bytes) => String::from_utf8(bytes)
            .map(Some)
            .map_err(|_| unexpected_reply("enqueued-token record is not UTF-8")),
        Value::SimpleString(text) => Ok(Some(text)),
        _ => Err(unexpected_reply(
            "enqueued-token record pop reply is not a string",
        )),
    }
}
