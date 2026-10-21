use crate::{
    Error,
    evidence::{MAX_TOOL_BYTES, limited},
    wire,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;
use tokio::io::AsyncReadExt;

#[derive(Serialize, Deserialize)]
pub struct Turn {
    pub response: String,
    pub tool_results: Vec<String>,
    pub validation_error: String,
}

pub struct Journal {
    directory: tempfile::TempDir,
    pub turns: Vec<usize>,
    bytes: usize,
}

struct Excerpt {
    start: usize,
    end: usize,
    characters: usize,
    text: String,
}

impl Excerpt {
    fn append(&mut self, text: &str) -> Result<(), Error> {
        let length = text.chars().count();
        let start = self.start.saturating_sub(self.characters);
        let end = self.end.saturating_sub(self.characters).min(length);
        if start < end {
            for character in text.chars().skip(start).take(end - start) {
                if self.text.len() + character.len_utf8() > MAX_TOOL_BYTES {
                    return Err(Error::ToolOutputTooLarge);
                }
                self.text.push(character);
            }
        }
        self.characters += length;
        Ok(())
    }

    async fn append_file(&mut self, path: &Path) -> Result<(), Error> {
        let mut file = tokio::fs::File::open(path).await?;
        let mut buffer = [0u8; 64 * 1024];
        let mut pending = Vec::new();
        loop {
            let count = file.read(&mut buffer).await?;
            if count == 0 {
                return if pending.is_empty() {
                    Ok(())
                } else {
                    Err(Error::InvalidRequest)
                };
            }
            pending.extend_from_slice(&buffer[..count]);
            let valid = match std::str::from_utf8(&pending) {
                Ok(_) => pending.len(),
                Err(error) if error.error_len().is_none() => error.valid_up_to(),
                Err(_) => return Err(Error::InvalidRequest),
            };
            self.append(
                std::str::from_utf8(&pending[..valid]).map_err(|_| Error::InvalidRequest)?,
            )?;
            pending.drain(..valid);
        }
    }
}

impl Journal {
    pub async fn new(initial: &Value) -> Result<Self, Error> {
        let directory = tempfile::Builder::new().prefix("lens-journal-").tempdir()?;
        let bytes = serde_json::to_vec(initial)?;
        tokio::fs::write(directory.path().join("initial"), &bytes).await?;
        Ok(Self {
            directory,
            turns: Vec::new(),
            bytes: bytes.len(),
        })
    }

    pub async fn push(&mut self, turn: &Turn) -> Result<(), Error> {
        let encoded = serde_json::to_string(turn)?;
        self.bytes += encoded.len();
        if self.bytes > 512 * 1024 * 1024 {
            return Err(Error::JournalTooLarge);
        }
        tokio::fs::write(
            self.directory.path().join(self.turns.len().to_string()),
            encoded.as_bytes(),
        )
        .await?;
        self.turns.push(encoded.chars().count());
        Ok(())
    }

    pub async fn reply(&self, request: &wire::EvidenceRequest) -> Result<Value, Error> {
        let start = request.turn_start as usize;
        let end = request
            .turn_end
            .map(|n| n as usize)
            .unwrap_or(self.turns.len())
            .min(self.turns.len());
        if start > end || request.char_end.is_some_and(|end| end < request.char_start) {
            return Ok(
                json!({"request": request, "error": "Choose a valid journal turn and character range"}),
            );
        }
        if request.char_start != 0 || request.char_end.is_some() {
            return self.excerpt(request, start, end).await;
        }
        let mut turns = Vec::<Value>::new();
        let mut bytes = 0;
        for index in start..end {
            let path = self.directory.path().join(index.to_string());
            bytes += tokio::fs::metadata(&path).await?.len();
            if bytes > 32 * 1024 * 1024 {
                return Err(Error::HistoryTooLarge);
            }
            turns.push(serde_json::from_slice(&tokio::fs::read(path).await?)?);
        }
        let initial: Value = if request.include_initial {
            serde_json::from_slice(&tokio::fs::read(self.directory.path().join("initial")).await?)?
        } else {
            Value::Null
        };
        let mut normalized = request.clone();
        normalized.char_start = 0;
        normalized.char_end = None;
        let reply = json!({"request": normalized, "total_turns": self.turns.len(), "initial_context": initial, "turns": turns, "turn_characters": self.turns});
        limited(reply)
    }

    async fn excerpt(
        &self,
        request: &wire::EvidenceRequest,
        start: usize,
        end: usize,
    ) -> Result<Value, Error> {
        let mut normalized = request.clone();
        normalized.char_start = 0;
        normalized.char_end = None;
        let document = json!({"request": normalized, "total_turns": self.turns.len(), "initial_context": null, "turns": [], "turn_characters": self.turns});
        let mut excerpt = Excerpt {
            start: request.char_start as usize,
            end: request
                .char_end
                .map(|value| value as usize)
                .unwrap_or(usize::MAX),
            characters: 0,
            text: String::new(),
        };
        excerpt.append("{")?;
        for (index, (key, value)) in document
            .as_object()
            .ok_or(Error::InvalidRequest)?
            .iter()
            .enumerate()
        {
            if index != 0 {
                excerpt.append(",")?;
            }
            excerpt.append(&serde_json::to_string(key)?)?;
            excerpt.append(":")?;
            match key.as_str() {
                "initial_context" if request.include_initial => {
                    excerpt
                        .append_file(&self.directory.path().join("initial"))
                        .await?;
                }
                "turns" => {
                    excerpt.append("[")?;
                    for turn in start..end {
                        if turn != start {
                            excerpt.append(",")?;
                        }
                        excerpt
                            .append_file(&self.directory.path().join(turn.to_string()))
                            .await?;
                    }
                    excerpt.append("]")?;
                }
                _ => excerpt.append(&serde_json::to_string(value)?)?,
            }
        }
        excerpt.append("}")?;
        limited(
            json!({"request": request, "total_turns": self.turns.len(), "excerpt": excerpt.text, "characters": excerpt.characters}),
        )
    }

    pub fn reference(&self, request: &wire::EvidenceRequest) -> Option<String> {
        if request.action != wire::EvidenceRequestAction::History
            || request.char_start != 0
            || request.char_end.is_some()
            || request.turn_start as usize > self.turns.len()
            || request.turn_end.is_some_and(|n| n < request.turn_start)
        {
            return None;
        }
        let mut request = request.clone();
        request.turn_end = Some(
            request
                .turn_end
                .unwrap_or(self.turns.len() as u64)
                .min(self.turns.len() as u64),
        );
        Some(json!({"kind": "history_reference", "request": request, "recorded_turns": self.turns.len()}).to_string())
    }
}
