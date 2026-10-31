use std::time::Duration;

use litellm_python_compat::json::dumps_str;
use litellm_redis_client::RedisClient;
use serde_json::{Map, Value};

use crate::Error;

const KEY_PREFIX: &str = "litellm:polling:response:";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State(String);

impl State {
    pub fn parse(document: String) -> Result<Self, serde_json::Error> {
        serde_json::from_str::<Map<String, Value>>(&document)?;
        Ok(Self(document))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn key(polling_id: &str) -> String {
    format!("{KEY_PREFIX}{polling_id}")
}

pub async fn put(
    client: &RedisClient,
    polling_id: &str,
    state: &State,
    ttl: Duration,
) -> Result<(), Error> {
    Ok(client
        .set(&key(polling_id), dumps_str(state.as_str()), ttl)
        .await?)
}

pub async fn get(client: &RedisClient, polling_id: &str) -> Result<Option<State>, Error> {
    let Some(raw) = client.get::<Vec<u8>>(&key(polling_id)).await? else {
        return Ok(None);
    };
    let document: String = serde_json::from_slice(&raw).map_err(Error::format)?;
    State::parse(document).map(Some).map_err(Error::format)
}

pub async fn delete(client: &RedisClient, polling_id: &str) -> Result<(), Error> {
    client.del(&[&key(polling_id)]).await?;
    Ok(())
}
