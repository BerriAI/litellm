#![allow(dead_code)]

use std::{
    collections::BTreeMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_cache::{CacheCodec, Error, JsonCodec};
use litellm_cache_redis::{RedisCache, RedisNode, RedisTopology};

/// Encodes a byte behind a tag, so a value written with another tag decodes as invalid.
pub struct TaggedByteCodec(pub u8);

impl CacheCodec for TaggedByteCodec {
    type Value = u8;

    fn encode(&self, value: &u8) -> Result<Vec<u8>, Error> {
        if *value > 127 {
            return Err(Error::InvalidEntry);
        }
        Ok(vec![self.0, *value])
    }

    fn decode(&self, bytes: &[u8]) -> Result<u8, Error> {
        match bytes {
            [tag, value] if *tag == self.0 => Ok(*value),
            _ => Err(Error::InvalidEntry),
        }
    }
}

/// A stateful in-process stand-in for a Redis server that understands the string commands the
/// shared contracts exercise, so they run without a live server. TTLs are accepted and ignored.
#[derive(Default)]
pub struct FakeRedis {
    strings: BTreeMap<Vec<u8>, Vec<u8>>,
}

impl FakeRedis {
    fn run(&mut self, command: Vec<Vec<u8>>) -> redis::RedisResult<redis::Value> {
        let name = String::from_utf8_lossy(&command[0]).to_ascii_uppercase();
        let args = &command[1..];
        Ok(match name.as_str() {
            "PING" => redis::Value::SimpleString("PONG".into()),
            "SET" | "SETEX" => {
                let value = if name == "SET" { &args[1] } else { &args[2] };
                self.strings.insert(args[0].clone(), value.clone());
                redis::Value::Okay
            }
            "GET" => self.get(&args[0]),
            "MGET" => redis::Value::Array(args.iter().map(|key| self.get(key)).collect()),
            "DEL" => {
                let removed = args
                    .iter()
                    .filter(|key| self.strings.remove(*key).is_some())
                    .count();
                redis::Value::Int(removed as i64)
            }
            "SCAN" => {
                let pattern = &args[2];
                let keys = self
                    .strings
                    .keys()
                    .filter(|key| glob(pattern, key))
                    .map(|key| redis::Value::BulkString(key.clone()))
                    .collect();
                redis::Value::Array(vec![
                    redis::Value::BulkString(b"0".to_vec()),
                    redis::Value::Array(keys),
                ])
            }
            "EVAL" if args[0].windows(11).any(|window| window == b"INCRBYFLOAT") => {
                self.increment_by_float(&args[2], &args[3])
            }
            "INCRBYFLOAT" => self.increment_by_float(&args[0], &args[1]),
            _ => {
                return Err(redis::RedisError::from((
                    redis::ErrorKind::Client,
                    "unsupported command",
                    name,
                )));
            }
        })
    }

    fn get(&self, key: &[u8]) -> redis::Value {
        self.strings.get(key).map_or(redis::Value::Nil, |value| {
            redis::Value::BulkString(value.clone())
        })
    }

    fn increment_by_float(&mut self, key: &[u8], amount: &[u8]) -> redis::Value {
        let current = self
            .strings
            .get(key)
            .map_or(0.0, |value| parse_float(value));
        let total = format!("{}", current + parse_float(amount));
        self.strings
            .insert(key.to_vec(), total.clone().into_bytes());
        redis::Value::BulkString(total.into_bytes())
    }
}

fn parse_float(bytes: &[u8]) -> f64 {
    std::str::from_utf8(bytes).unwrap().parse().unwrap()
}

/// Redis `MATCH` globbing for `*`, `?` and backslash escapes.
fn glob(pattern: &[u8], key: &[u8]) -> bool {
    match pattern.split_first() {
        None => key.is_empty(),
        Some((b'*', rest)) => (0..=key.len()).any(|skip| glob(rest, &key[skip..])),
        Some((b'?', rest)) => !key.is_empty() && glob(rest, &key[1..]),
        Some((b'\\', [escaped, rest @ ..])) => {
            key.first() == Some(escaped) && glob(rest, &key[1..])
        }
        Some((literal, rest)) => key.first() == Some(literal) && glob(rest, &key[1..]),
    }
}

/// Splits RESP request bytes into the commands they carry.
fn commands(mut bytes: &[u8]) -> Vec<Vec<Vec<u8>>> {
    fn line<'a>(bytes: &mut &'a [u8]) -> &'a [u8] {
        let end = bytes
            .windows(2)
            .position(|window| window == b"\r\n")
            .unwrap();
        let (line, rest) = bytes.split_at(end);
        *bytes = &rest[2..];
        line
    }
    fn length(line: &[u8]) -> usize {
        std::str::from_utf8(&line[1..]).unwrap().parse().unwrap()
    }
    let mut commands = Vec::new();
    while !bytes.is_empty() {
        let count = length(line(&mut bytes));
        let command = (0..count)
            .map(|_| {
                let size = length(line(&mut bytes));
                let (argument, rest) = bytes.split_at(size);
                bytes = &rest[2..];
                argument.to_vec()
            })
            .collect();
        commands.push(command);
    }
    commands
}

impl redis::ConnectionLike for FakeRedis {
    fn req_packed_command(&mut self, cmd: &[u8]) -> redis::RedisResult<redis::Value> {
        let command = commands(cmd).into_iter().next().unwrap();
        self.run(command)
    }

    fn req_packed_commands(
        &mut self,
        cmd: &[u8],
        offset: usize,
        count: usize,
    ) -> redis::RedisResult<Vec<redis::Value>> {
        let replies = commands(cmd)
            .into_iter()
            .map(|command| self.run(command))
            .collect::<redis::RedisResult<Vec<_>>>()?;
        Ok(replies.into_iter().skip(offset).take(count).collect())
    }

    fn get_db(&self) -> i64 {
        0
    }

    fn check_connection(&mut self) -> bool {
        true
    }

    fn is_open(&self) -> bool {
        true
    }
}

pub type JsonCache<C = redis::Connection> = RedisCache<JsonCodec<serde_json::Value>, C>;

pub fn fake_cache(namespace: &str) -> JsonCache<FakeRedis> {
    RedisCache::with_connection(FakeRedis::default(), None, JsonCodec::new())
        .with_namespace(Some(namespace.into()))
}

/// Startup nodes from `LITELLM_TEST_REDIS_CLUSTER_NODES` (`host:port,host:port`); tests that
/// need a live cluster skip when it is unset.
pub fn cluster_topology() -> Option<RedisTopology> {
    let nodes = std::env::var("LITELLM_TEST_REDIS_CLUSTER_NODES").ok()?;
    let startup_nodes = nodes
        .split(',')
        .map(|node| {
            let (host, port) = node.trim().rsplit_once(':').expect("host:port");
            RedisNode {
                host: host.to_string(),
                port: port.parse().expect("port"),
            }
        })
        .collect();
    Some(RedisTopology::Cluster { startup_nodes })
}

pub fn cluster_url() -> String {
    std::env::var("LITELLM_TEST_REDIS_CLUSTER_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:7000".into())
}

pub fn unique_namespace(label: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("cluster-test:{label}:{nanos}")
}

pub fn cluster_cache<S: CacheCodec>(
    label: &str,
    default_ttl: Duration,
    codec: S,
) -> Option<RedisCache<S>> {
    let topology = cluster_topology()?;
    Some(
        RedisCache::connect(&cluster_url(), &topology, Some(default_ttl), codec)
            .expect("cluster connection")
            .with_namespace(Some(unique_namespace(label))),
    )
}
