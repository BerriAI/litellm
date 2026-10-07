use crate::{
    Error,
    evidence::{character_range, limited},
    wire,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

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
        if request.char_start != 0 || request.char_end.is_some() {
            let serialized = serde_json::to_string(&reply)?;
            return limited(
                json!({"request": request, "total_turns": self.turns.len(), "excerpt": character_range(&serialized, request.char_start as usize, request.char_end.map(|n| n as usize)), "characters": serialized.chars().count()}),
            );
        }
        limited(reply)
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
