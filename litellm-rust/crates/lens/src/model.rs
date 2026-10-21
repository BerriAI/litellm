use crate::{Error, control::JobClient, wire};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, VecDeque},
    sync::OnceLock,
};

pub fn schema(name: &str) -> Result<Value, Error> {
    static CONTRACT: OnceLock<Value> = OnceLock::new();
    let contract = CONTRACT.get_or_init(|| {
        serde_json::from_str(include_str!("../contract.json")).expect("validated at build time")
    });
    let definitions = contract["definitions"]
        .as_object()
        .ok_or(Error::InvalidRequest)?;
    let mut root = definitions
        .get(name)
        .cloned()
        .ok_or(Error::InvalidRequest)?;
    let mut pending = VecDeque::new();
    references(&root, &mut pending);
    let mut selected = serde_json::Map::new();
    let mut seen = BTreeSet::new();
    while let Some(name) = pending.pop_front() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let definition = definitions.get(&name).ok_or(Error::InvalidRequest)?;
        references(definition, &mut pending);
        selected.insert(name, definition.clone());
    }
    root.as_object_mut()
        .ok_or(Error::InvalidRequest)?
        .insert("definitions".into(), selected.into());
    Ok(root)
}

fn references(value: &Value, found: &mut VecDeque<String>) {
    match value {
        Value::Object(object) => {
            if let Some(reference) = object
                .get("$ref")
                .and_then(Value::as_str)
                .and_then(|s| s.strip_prefix("#/definitions/"))
            {
                found.push_back(reference.into());
            }
            for value in object.values() {
                references(value, found);
            }
        }
        Value::Array(values) => {
            for value in values {
                references(value, found);
            }
        }
        _ => {}
    }
}

pub fn message(role: wire::ModelMessageRole, content: impl Into<String>) -> wire::ModelMessage {
    wire::ModelMessage {
        role,
        content: content.into(),
    }
}

pub fn request(
    purpose: wire::ModelRequestPurpose,
    prompt: Value,
) -> Result<wire::ModelRequest, Error> {
    Ok(wire::ModelRequest {
        purpose,
        messages: Vec::new(),
        prompt: serde_json::to_string(&prompt)?
            .try_into()
            .map_err(|_| Error::InvalidRequest)?,
    })
}

pub async fn structured<T: DeserializeOwned>(
    client: &JobClient,
    mut request: wire::ModelRequest,
    schema_name: &'static str,
    validate: impl Fn(&T) -> Option<String>,
) -> Result<(T, Vec<wire::ModelMessage>), Error> {
    let validator =
        jsonschema::validator_for(&schema(schema_name)?).map_err(|_| Error::InvalidRequest)?;
    let mut detail = String::new();
    for attempt in 0..2 {
        let response = client.model(&request).await?;
        if response.context_exceeded {
            return Err(Error::Context(Box::new(request)));
        }
        let value: Result<Value, _> = serde_json::from_str(&response.content);
        let contract_error = value
            .as_ref()
            .ok()
            .and_then(|value| validator.validate(value).err())
            .map(|error| error.to_string());
        let parsed: Result<T, _> = value.and_then(serde_json::from_value);
        detail = match parsed {
            Ok(ref value) if response.finish_reason.is_none() => contract_error
                .or_else(|| validate(value))
                .unwrap_or_default(),
            Ok(_) => "Model did not finish its response. Return a complete JSON object.".into(),
            Err(ref error) => error.to_string(),
        };
        if detail.is_empty() {
            request
                .messages
                .push(message(wire::ModelMessageRole::Assistant, response.content));
            return Ok((parsed?, request.messages));
        }
        if attempt == 0 {
            if request.messages.is_empty() {
                request.messages.push(message(
                    wire::ModelMessageRole::User,
                    request.prompt.to_string(),
                ));
            }
            request
                .messages
                .push(message(wire::ModelMessageRole::Assistant, response.content));
            request.messages.push(message(wire::ModelMessageRole::System, json!({
                "instruction": "Your previous response did not match the required response contract. Generate a new response from the original evidence, correcting the validation errors. Follow the complete object structure in response_schema. If the schema allows tools, you may request them before finalizing.",
                "validation_errors": detail,
                "response_schema": schema(schema_name)?,
            }).to_string()));
        }
    }
    Err(Error::ModelValidation {
        schema: schema_name,
        detail,
    })
}

fn visible_journal(messages: &[wire::ModelMessage]) -> usize {
    let positions: Vec<Value> = messages
        .iter()
        .filter(|m| m.role == wire::ModelMessageRole::User)
        .filter_map(|m| serde_json::from_str(&m.content).ok())
        .collect();
    let visible = positions
        .iter()
        .filter_map(|p| p["journal_turns"].as_u64())
        .max()
        .unwrap_or_default();
    positions
        .iter()
        .filter_map(|p| p["resume_history_from_turn"].as_u64())
        .min()
        .unwrap_or(visible) as usize
}

pub async fn compact(
    client: &JobClient,
    mut request: wire::ModelRequest,
    journal_turns: usize,
) -> Result<Vec<wire::ModelMessage>, Error> {
    let instruction = message(wire::ModelMessageRole::System, json!({ "task": include_str!("../prompts/compact.md"), "response_schema": schema("Checkpoint")? }).to_string());
    if request.messages.is_empty() {
        request.messages.push(message(
            wire::ModelMessageRole::System,
            request.prompt.to_string(),
        ));
    }
    loop {
        let mut summarize = request.clone();
        summarize.messages.push(instruction.clone());
        match structured::<wire::Checkpoint>(client, summarize, "Checkpoint", |_| None).await {
            Ok((notes, _)) => {
                return Ok(vec![
                    request.messages[0].clone(),
                    message(
                        wire::ModelMessageRole::User,
                        json!({
                            "working_notes": notes.working_notes,
                            "journal_turns": journal_turns,
                            "resume_history_from_turn": visible_journal(&request.messages),
                            "initial_context_archived": true,
                        })
                        .to_string(),
                    ),
                ]);
            }
            Err(Error::Context(_)) if request.messages.len() > 1 => {
                request
                    .messages
                    .truncate((request.messages.len() / 2).max(1));
                if request.messages.len() > 1
                    && request
                        .messages
                        .last()
                        .is_some_and(|m| m.role == wire::ModelMessageRole::Assistant)
                {
                    request.messages.pop();
                }
            }
            Err(Error::Context(_)) => {
                return Err(Error::TaskContext);
            }
            Err(error) => return Err(error),
        }
    }
}
