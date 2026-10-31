use std::time::Duration;

use litellm_python_compat::{Value, json::to_json, literal::literal_eval, repr::repr};
use litellm_redis_client::RedisClient;
use serde::Deserialize;

use crate::Error;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Cooldown {
    pub exception_received: String,
    pub status_code: String,
    pub timestamp: f64,
    pub cooldown_time: f64,
}

fn key(model_id: &str) -> String {
    format!("deployment:{model_id}:cooldown")
}

pub async fn put(
    client: &RedisClient,
    model_id: &str,
    cooldown: &Cooldown,
    ttl: Duration,
) -> Result<(), Error> {
    Ok(client.set(&key(model_id), encode(cooldown), ttl).await?)
}

pub async fn get_many(
    client: &RedisClient,
    model_ids: &[&str],
) -> Result<Vec<Option<Result<Cooldown, Error>>>, Error> {
    let keys: Vec<String> = model_ids.iter().map(|model_id| key(model_id)).collect();
    let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
    let stored: Vec<Option<Vec<u8>>> = client.mget(&keys).await?;
    Ok(stored
        .into_iter()
        .map(|raw| raw.map(|raw| decode(&raw)))
        .collect())
}

fn encode(cooldown: &Cooldown) -> String {
    let field = |name: &str, value: Value| (Value::Str(name.to_owned()), value);
    repr(&Value::Dict(vec![
        field(
            "exception_received",
            Value::Str(cooldown.exception_received.clone()),
        ),
        field("status_code", Value::Str(cooldown.status_code.clone())),
        field("timestamp", Value::Float(cooldown.timestamp)),
        field("cooldown_time", Value::Float(cooldown.cooldown_time)),
    ]))
}

fn decode(raw: &[u8]) -> Result<Cooldown, Error> {
    let text = std::str::from_utf8(raw).map_err(Error::format)?;
    let document = match serde_json::from_str(text) {
        Ok(document) => document,
        Err(_) => to_json(&literal_eval(text).map_err(Error::format)?).map_err(Error::format)?,
    };
    serde_json::from_value(document).map_err(Error::format)
}
