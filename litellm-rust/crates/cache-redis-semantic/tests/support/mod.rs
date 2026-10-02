#![allow(dead_code)]

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
};

use litellm_cache::{Error, semantic::Embedder};
use serde_json::Value;

pub type EmbedCalls = Arc<Mutex<Vec<(String, Option<Value>)>>>;

/// Embeds known prompts to fixed vectors, anything else to `[0.1, 0.2, 0.3]`, and records every
/// prompt with its metadata.
pub struct FakeEmbedder {
    vectors: HashMap<String, Vec<f32>>,
    pub calls: EmbedCalls,
}

impl FakeEmbedder {
    pub fn new(vectors: &[(&str, &[f32])]) -> Self {
        Self {
            vectors: vectors
                .iter()
                .map(|(prompt, vector)| ((*prompt).to_owned(), vector.to_vec()))
                .collect(),
            calls: EmbedCalls::default(),
        }
    }
}

impl Embedder for FakeEmbedder {
    fn embed(&self, prompt: &str, metadata: Option<&Value>) -> Result<Vec<f32>, Error> {
        self.calls
            .lock()
            .unwrap()
            .push((prompt.to_owned(), metadata.cloned()));
        Ok(self
            .vectors
            .get(prompt)
            .cloned()
            .unwrap_or_else(|| vec![0.1, 0.2, 0.3]))
    }

    async fn async_embed(&self, prompt: &str, metadata: Option<&Value>) -> Result<Vec<f32>, Error> {
        self.embed(prompt, metadata)
    }
}

struct FakeIndex {
    prefix: Vec<u8>,
    dims: usize,
    vector_field: String,
}

#[derive(Default)]
struct SearchState {
    indexes: HashMap<String, FakeIndex>,
    hashes: BTreeMap<Vec<u8>, BTreeMap<String, Vec<u8>>>,
}

/// An in-memory Redis Stack speaking the `FT.*`, `HSET` and `EXPIRE` subset the semantic cache
/// sends, with exact cosine KNN over the hashes under an index prefix.
#[derive(Clone, Default)]
pub struct FakeSearch {
    state: Arc<Mutex<SearchState>>,
}

impl FakeSearch {
    fn run(&self, args: Vec<Vec<u8>>) -> redis::RedisResult<redis::Value> {
        let mut state = self.state.lock().unwrap();
        let text = |index: usize| String::from_utf8_lossy(&args[index]).into_owned();
        match text(0).to_uppercase().as_str() {
            "FT.CREATE" => {
                let name = text(1);
                if state.indexes.contains_key(&name) {
                    return Err(error("Index already exists"));
                }
                let position = |token: &str| args.iter().position(|arg| arg == token.as_bytes());
                let prefix = args[position("PREFIX").unwrap() + 2].clone();
                let dims = text(position("DIM").unwrap() + 1).parse().unwrap();
                let vector_field = text(position("VECTOR").unwrap() - 1);
                state.indexes.insert(
                    name,
                    FakeIndex {
                        prefix,
                        dims,
                        vector_field,
                    },
                );
                Ok(redis::Value::Okay)
            }
            "FT.INFO" => {
                let index = state
                    .indexes
                    .get(&text(1))
                    .ok_or_else(|| error("Unknown index name"))?;
                Ok(index_info(index))
            }
            "FT.DROPINDEX" => {
                state.indexes.remove(&text(1));
                Ok(redis::Value::Okay)
            }
            "HSET" => {
                let hash = state.hashes.entry(args[1].clone()).or_default();
                for pair in args[2..].chunks(2) {
                    hash.insert(
                        String::from_utf8_lossy(&pair[0]).into_owned(),
                        pair[1].clone(),
                    );
                }
                Ok(redis::Value::Int(((args.len() - 2) / 2) as i64))
            }
            "EXPIRE" => Ok(redis::Value::Int(i64::from(
                state.hashes.contains_key(&args[1]),
            ))),
            "FT.SEARCH" => {
                let index = state
                    .indexes
                    .get(&text(1))
                    .ok_or_else(|| error("no such index"))?;
                let query = text(2);
                let tag = query_tag(&query);
                let params = args.iter().position(|arg| arg == b"PARAMS").unwrap();
                let vector = floats(&args[params + 3]);
                let best = state
                    .hashes
                    .iter()
                    .filter(|(key, _)| key.starts_with(&index.prefix))
                    .filter(|(_, fields)| {
                        fields.get("litellm_cache_key").map(Vec::as_slice) == Some(tag.as_bytes())
                    })
                    .filter_map(|(key, fields)| {
                        let stored = floats(fields.get(&index.vector_field)?);
                        (stored.len() == index.dims)
                            .then(|| (key, fields, 1.0 - cosine(&vector, &stored)))
                    })
                    .min_by(|left, right| left.2.total_cmp(&right.2));
                let Some((key, fields, distance)) = best else {
                    return Ok(redis::Value::Array(vec![redis::Value::Int(0)]));
                };
                let mut reply = fields
                    .iter()
                    .filter(|(name, _)| **name != index.vector_field)
                    .flat_map(|(name, value)| [bulk(name.as_bytes()), bulk(value)])
                    .collect::<Vec<_>>();
                reply.extend([
                    bulk(b"vector_distance"),
                    bulk(distance.to_string().as_bytes()),
                ]);
                Ok(redis::Value::Array(vec![
                    redis::Value::Int(1),
                    bulk(key),
                    redis::Value::Array(reply),
                ]))
            }
            "PING" => Ok(redis::Value::SimpleString("PONG".into())),
            _ => Err(error("unsupported command")),
        }
    }
}

impl redis::ConnectionLike for FakeSearch {
    fn req_packed_command(&mut self, command: &[u8]) -> redis::RedisResult<redis::Value> {
        let mut commands = parse_commands(command);
        self.run(commands.remove(0))
    }

    fn req_packed_commands(
        &mut self,
        commands: &[u8],
        offset: usize,
        count: usize,
    ) -> redis::RedisResult<Vec<redis::Value>> {
        let replies = parse_commands(commands)
            .into_iter()
            .map(|args| self.run(args))
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

fn error(message: &'static str) -> redis::RedisError {
    redis::RedisError::from((redis::ErrorKind::Extension, message))
}

fn bulk(bytes: &[u8]) -> redis::Value {
    redis::Value::BulkString(bytes.to_vec())
}

fn index_info(index: &FakeIndex) -> redis::Value {
    let attribute = |name: &str, field_type: &str| {
        redis::Value::Array(vec![
            bulk(b"identifier"),
            bulk(name.as_bytes()),
            bulk(b"type"),
            bulk(field_type.as_bytes()),
        ])
    };
    redis::Value::Array(vec![
        bulk(b"attributes"),
        redis::Value::Array(vec![
            attribute("prompt", "TEXT"),
            attribute("response", "TEXT"),
            attribute("inserted_at", "NUMERIC"),
            attribute("updated_at", "NUMERIC"),
            attribute("litellm_cache_key", "TAG"),
            redis::Value::Array(vec![
                bulk(b"identifier"),
                bulk(index.vector_field.as_bytes()),
                bulk(b"type"),
                bulk(b"VECTOR"),
                bulk(b"dim"),
                redis::Value::Int(index.dims as i64),
                bulk(b"data_type"),
                bulk(b"FLOAT32"),
                bulk(b"distance_metric"),
                bulk(b"COSINE"),
            ]),
        ]),
    ])
}

/// The tag inside `@litellm_cache_key:{...}`, with query escapes removed.
fn query_tag(query: &str) -> String {
    let start = query.find("@litellm_cache_key:{").unwrap() + "@litellm_cache_key:{".len();
    let mut tag = String::new();
    let mut characters = query[start..].chars();
    while let Some(character) = characters.next() {
        match character {
            '\\' => tag.extend(characters.next()),
            '}' => break,
            character => tag.push(character),
        }
    }
    tag
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum::<f64>();
    let norm = |vector: &[f32]| {
        vector
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    dot / (norm(left) * norm(right))
}

/// Splits a packed RESP request into each command's arguments.
fn parse_commands(mut bytes: &[u8]) -> Vec<Vec<Vec<u8>>> {
    let line = |bytes: &mut &[u8]| {
        let end = bytes
            .windows(2)
            .position(|window| window == b"\r\n")
            .unwrap();
        let text = String::from_utf8(bytes[1..end].to_vec()).unwrap();
        *bytes = &bytes[end + 2..];
        text.parse::<usize>().unwrap()
    };
    let mut commands = Vec::new();
    while !bytes.is_empty() {
        let count = line(&mut bytes);
        let mut args = Vec::with_capacity(count);
        for _ in 0..count {
            let length = line(&mut bytes);
            args.push(bytes[..length].to_vec());
            bytes = &bytes[length + 2..];
        }
        commands.push(args);
    }
    commands
}
