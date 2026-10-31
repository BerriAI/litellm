use std::{sync::LazyLock, time::Duration};

use litellm_python_compat::{Value as PyValue, json::dumps, repr::float_repr};
use litellm_redis_client::{RedisClient, Script, Value};

use crate::{Error, as_strs};

static SET_MAX: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        "local cur = redis.call('GET', KEYS[1]) \
         if cur == false or tonumber(cur) < tonumber(ARGV[1]) then \
         redis.call('SET', KEYS[1], ARGV[1]) \
         if tonumber(ARGV[2]) > 0 then redis.call('EXPIRE', KEYS[1], ARGV[2]) end \
         return ARGV[1] end \
         return cur",
    )
});

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Counter {
    Key(String),
    Team(String),
    TeamMember { user_id: String, team_id: String },
    User(String),
    EndUser(String),
    Org(String),
    Tag(String),
    Project(String),
    KeyWindow { token: String, duration: String },
    TeamWindow { team_id: String, duration: String },
}

impl Counter {
    fn key(&self) -> String {
        match self {
            Self::Key(token) => format!("spend:key:{token}"),
            Self::Team(team_id) => format!("spend:team:{team_id}"),
            Self::TeamMember { user_id, team_id } => {
                format!("spend:team_member:{user_id}:{team_id}")
            }
            Self::User(user_id) => format!("spend:user:{user_id}"),
            Self::EndUser(end_user_id) => format!("spend:end_user:{end_user_id}"),
            Self::Org(org_id) => format!("spend:org:{org_id}"),
            Self::Tag(tag) => format!("spend:tag:{tag}"),
            Self::Project(project_id) => format!("spend:project:{project_id}"),
            Self::KeyWindow { token, duration } => format!("spend:key:{token}:window:{duration}"),
            Self::TeamWindow { team_id, duration } => {
                format!("spend:team:{team_id}:window:{duration}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Seed {
    Seeded,
    Existing(Option<f64>),
}

pub async fn read(client: &RedisClient, counters: &[Counter]) -> Result<Vec<Option<f64>>, Error> {
    let keys: Vec<String> = counters.iter().map(Counter::key).collect();
    let stored: Vec<Option<Vec<u8>>> = client.mget(&as_strs(&keys)).await?;
    stored
        .iter()
        .map(|raw| raw.as_deref().map(decode).transpose())
        .collect()
}

pub async fn add(
    client: &RedisClient,
    deltas: &[(Counter, f64)],
    ttl: Duration,
) -> Result<Vec<f64>, Error> {
    let pipeline = deltas
        .iter()
        .fold(client.pipeline(), |pipeline, (counter, delta)| {
            let key = counter.key();
            pipeline.incr_by_float(&key, *delta).expire(&key, ttl)
        });
    client
        .run(&pipeline)
        .await
        .into_iter()
        .step_by(2)
        .map(|reply| total(reply?))
        .collect()
}

pub async fn seed(
    client: &RedisClient,
    counter: &Counter,
    spend: f64,
    ttl: Duration,
) -> Result<Seed, Error> {
    let encoded = dumps(&PyValue::Float(spend)).map_err(Error::format)?;
    if client.set_nx(&counter.key(), encoded, ttl).await? {
        return Ok(Seed::Seeded);
    }
    let stored = read(client, std::slice::from_ref(counter)).await?;
    Ok(Seed::Existing(stored.into_iter().next().flatten()))
}

pub async fn raise_to(
    client: &RedisClient,
    counter: &Counter,
    spend: f64,
    ttl: Duration,
) -> Result<f64, Error> {
    let stored: Vec<u8> = client
        .eval(
            &SET_MAX,
            &[&counter.key()],
            vec![float_repr(spend), ttl.as_secs().to_string()],
        )
        .await?;
    decode(&stored)
}

fn decode(raw: &[u8]) -> Result<f64, Error> {
    std::str::from_utf8(raw)
        .map_err(Error::format)?
        .parse()
        .map_err(Error::format)
}

fn total(reply: Value) -> Result<f64, Error> {
    match reply {
        Value::BulkString(raw) => decode(&raw),
        Value::Double(total) => Ok(total),
        reply => Err(Error::unexpected(reply)),
    }
}
