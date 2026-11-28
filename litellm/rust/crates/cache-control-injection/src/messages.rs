//! The injection points on a Messages request. Python's
//! `AnthropicCacheControlHook.apply_to_anthropic_messages_request`.
//!
//! [`inject`] is three steps over plain data. [`Census`] counts the breakpoints the request
//! already carries and finds where each message can take one, [`plan`] decides which points
//! are placed, and the request is rebuilt with them.
//!
//! - The provider accepts [`LIMIT`] breakpoints a request. The caller's own count first, on
//!   tools and the top-level `cache_control` as well, and a point is placed only while a slot
//!   is left. A caller's breakpoint is never moved or removed
//! - The system point goes on the last system block, and only when no system block is marked
//! - A message point goes on the last block of its message that takes a breakpoint, and only
//!   when the message carries none. Points are placed in the order the config lists them
//! - Every message's text content becomes one text block, placed on or not, as Python sends it
//!
//! A breakpoint on a chat-shaped `tool_calls` entry is not counted, since no Messages host
//! accepts that key.

use litellm_llms_types::{
    formats::messages::{
        CacheControl, ContentBlock, ContentBlockType, Message, MessageContent,
        MessagesOptionalParams, MessagesRequest, MessagesTool, SystemPrompt,
    },
    recognized::Recognized,
};
use serde_json::{Map, Value};

use crate::CacheControlInjectionPoint;

/// "A maximum of 4 blocks with cache_control may be provided."
const LIMIT: usize = 4;
const BREAKPOINT_KEYS: [&str; 2] = ["cache_control", "prompt_cache_breakpoint"];
const UNMARKED_BLOCK_TYPES: [&str; 2] = ["thinking", "redacted_thinking"];
const SYSTEM: &str = "system";

/// The injection points of one deployment that the Messages route places itself.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CachePoints {
    system: Option<CacheControl>,
    messages: Vec<MessagePoint>,
}

#[derive(Clone, Debug, PartialEq)]
struct MessagePoint {
    at: At,
    control: CacheControl,
}

#[derive(Clone, Debug, PartialEq)]
enum At {
    /// One message, counted from the end when negative.
    Index(i64),
    /// Every message of a role.
    Role(String),
}

impl CachePoints {
    /// The points as this route places them, or `None` when the config holds one it does not:
    /// a `tool_config` point, a message point that names neither a role nor an index, or a
    /// value Python's hook reads otherwise than as a point.
    pub fn served(
        written: &Recognized<Vec<Recognized<CacheControlInjectionPoint>>>,
    ) -> Option<Self> {
        let Recognized::Known(written) = written else {
            return None;
        };
        written
            .iter()
            .try_fold(Self::default(), |points, point| match point {
                Recognized::Known(CacheControlInjectionPoint::Message {
                    role: Some(role),
                    control,
                    ..
                }) if role == SYSTEM => Some(Self {
                    system: points.system.or_else(|| Some(control_or_default(control))),
                    ..points
                }),
                Recognized::Known(CacheControlInjectionPoint::Message {
                    index: Some(index),
                    control,
                    ..
                }) => Some(points.with(At::Index(*index), control)),
                Recognized::Known(CacheControlInjectionPoint::Message {
                    role: Some(role),
                    control,
                    ..
                }) => Some(points.with(At::Role(role.clone()), control)),
                _ => None,
            })
    }

    fn with(self, at: At, control: &Option<CacheControl>) -> Self {
        let point = MessagePoint {
            at,
            control: control_or_default(control),
        };
        Self {
            messages: self.messages.into_iter().chain([point]).collect(),
            ..self
        }
    }

    fn is_empty(&self) -> bool {
        self.system.is_none() && self.messages.is_empty()
    }
}

/// `point.get("control") or {"type": "ephemeral"}`.
fn control_or_default(control: &Option<CacheControl>) -> CacheControl {
    control
        .clone()
        .filter(|control| *control != CacheControl::default())
        .unwrap_or_else(|| CacheControl {
            cache_type: Some("ephemeral".to_owned()),
            ..CacheControl::default()
        })
}

/// Where one message or the system prompt stands before any point is placed.
#[derive(Debug, Default, PartialEq)]
struct Seat<'a> {
    role: Option<&'a str>,
    /// The breakpoints it carries.
    marks: usize,
    /// The block a breakpoint goes on, once its text content is a block.
    last: Option<usize>,
}

struct Census<'a> {
    system: Seat<'a>,
    messages: Vec<Seat<'a>>,
    /// The caller's breakpoints outside the system prompt and the messages.
    outside: usize,
}

/// The breakpoint each seat is given, on the block it goes on.
#[derive(Debug, Default, PartialEq)]
struct Plan {
    system: Option<(usize, CacheControl)>,
    messages: Vec<Option<(usize, CacheControl)>>,
}

fn is_set(fields: &Map<String, Value>, key: &str) -> bool {
    fields.get(key).is_some_and(|value| !value.is_null())
}

fn carries_breakpoint(fields: &Map<String, Value>) -> bool {
    BREAKPOINT_KEYS.iter().any(|key| is_set(fields, key))
}

fn block_is_marked(block: &ContentBlock) -> bool {
    matches!(block.cache_control, Some(Recognized::Known(_))) || carries_breakpoint(&block.extra)
}

fn raw_block_is_marked(block: &Value) -> bool {
    block.as_object().is_some_and(carries_breakpoint)
}

fn block_takes_a_breakpoint(block: &ContentBlock) -> bool {
    !matches!(
        block.known_type(),
        Some(ContentBlockType::Thinking | ContentBlockType::RedactedThinking)
    )
}

fn raw_block_takes_a_breakpoint(block: &Value) -> bool {
    block.as_object().is_some_and(|fields| {
        !fields
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| UNMARKED_BLOCK_TYPES.contains(&kind))
    })
}

impl<'a> Seat<'a> {
    fn of_message(message: &'a Recognized<Message>) -> Self {
        match message {
            Recognized::Known(message) => {
                let own = usize::from(carries_breakpoint(&message.extra));
                let role = Some(message.role.as_str());
                match &message.content {
                    MessageContent::Text(_) => Self {
                        role,
                        marks: own,
                        last: Some(0),
                    },
                    MessageContent::Blocks(blocks) => Self {
                        role,
                        marks: own + blocks.iter().filter(|block| block_is_marked(block)).count(),
                        last: blocks.iter().rposition(block_takes_a_breakpoint),
                    },
                }
            }
            Recognized::Unrecognized(Value::Object(fields)) => {
                let own = usize::from(carries_breakpoint(fields));
                let role = fields.get("role").and_then(Value::as_str);
                match fields.get("content") {
                    Some(Value::String(_)) => Self {
                        role,
                        marks: own,
                        last: Some(0),
                    },
                    Some(Value::Array(blocks)) => Self {
                        role,
                        marks: own
                            + blocks
                                .iter()
                                .filter(|block| raw_block_is_marked(block))
                                .count(),
                        last: blocks.iter().rposition(raw_block_takes_a_breakpoint),
                    },
                    _ => Self {
                        role,
                        marks: own,
                        last: None,
                    },
                }
            }
            Recognized::Unrecognized(_) => Self::default(),
        }
    }

    /// The last system block takes the breakpoint whatever its type, as long as it is an object.
    fn of_system(system: Option<&Recognized<SystemPrompt>>) -> Self {
        match system {
            Some(Recognized::Known(SystemPrompt::Text(_))) => Self {
                last: Some(0),
                ..Self::default()
            },
            Some(Recognized::Known(SystemPrompt::Blocks(blocks))) => Self {
                marks: blocks.iter().filter(|block| block_is_marked(block)).count(),
                last: blocks.len().checked_sub(1),
                ..Self::default()
            },
            Some(Recognized::Unrecognized(Value::Array(blocks))) => Self {
                marks: blocks
                    .iter()
                    .filter(|block| raw_block_is_marked(block))
                    .count(),
                last: blocks
                    .len()
                    .checked_sub(1)
                    .filter(|last| blocks[*last].is_object()),
                ..Self::default()
            },
            _ => Self::default(),
        }
    }
}

/// A tool carries its breakpoint at the top level, or under `function` in the OpenAI shape.
fn tool_is_marked(tool: &Recognized<MessagesTool>) -> bool {
    let fields = match tool {
        Recognized::Known(
            MessagesTool::Advisor { extra }
            | MessagesTool::ToolSearchRegex { extra }
            | MessagesTool::ToolSearchBm25 { extra },
        ) => Some(extra),
        Recognized::Unrecognized(tool) => tool.as_object(),
    };
    fields.is_some_and(|fields| {
        carries_breakpoint(fields)
            || fields
                .get("function")
                .and_then(Value::as_object)
                .is_some_and(carries_breakpoint)
    })
}

impl<'a> Census<'a> {
    fn of(request: &'a MessagesRequest) -> Self {
        let params = &request.params;
        let automatic = params
            .cache_control
            .as_ref()
            .is_some_and(|cache_control| !cache_control.is_null());
        Self {
            system: Seat::of_system(params.system.as_ref()),
            messages: request.messages.iter().map(Seat::of_message).collect(),
            outside: usize::from(automatic)
                + params
                    .tools
                    .iter()
                    .flatten()
                    .filter(|tool| tool_is_marked(tool))
                    .count(),
        }
    }
}

impl At {
    fn targets(&self, messages: &[Seat<'_>]) -> Vec<usize> {
        match self {
            Self::Index(index) => {
                let count = i64::try_from(messages.len()).unwrap_or(i64::MAX);
                let from_start = if *index < 0 { index + count } else { *index };
                usize::try_from(from_start)
                    .ok()
                    .filter(|target| *target < messages.len())
                    .into_iter()
                    .collect()
            }
            Self::Role(role) => messages
                .iter()
                .enumerate()
                .filter(|(_, seat)| seat.role == Some(role.as_str()))
                .map(|(target, _)| target)
                .collect(),
        }
    }
}

fn plan(points: &CachePoints, census: &Census<'_>) -> Plan {
    let limit = LIMIT.saturating_sub(census.outside);
    let in_messages: usize = census.messages.iter().map(|seat| seat.marks).sum();
    let system = points
        .system
        .as_ref()
        .filter(|_| census.system.marks == 0 && in_messages < limit)
        .and_then(|control| Some((census.system.last?, control.clone())));
    let budget = limit.saturating_sub(census.system.marks + usize::from(system.is_some()));
    let targets = points.messages.iter().flat_map(|point| {
        point
            .at
            .targets(&census.messages)
            .into_iter()
            .map(move |target| (target, &point.control))
    });
    let mut messages = vec![None; census.messages.len()];
    let mut used = in_messages;
    for (target, control) in targets {
        if used >= budget {
            break;
        }
        let seat = &census.messages[target];
        if let (0, None, Some(block)) = (seat.marks, &messages[target], seat.last) {
            messages[target] = Some((block, control.clone()));
            used += 1;
        }
    }
    Plan { system, messages }
}

fn marked(block: ContentBlock, control: CacheControl) -> ContentBlock {
    ContentBlock {
        cache_control: Some(Recognized::Known(control)),
        ..block
    }
}

fn raw_marked(block: Value, control: &CacheControl) -> Value {
    let Value::Object(fields) = block else {
        return block;
    };
    let control = serde_json::to_value(control).expect("a cache control serializes");
    Value::Object(
        fields
            .into_iter()
            .filter(|(key, _)| key != "cache_control")
            .chain([("cache_control".to_owned(), control)])
            .collect(),
    )
}

fn blocks_with(
    blocks: Vec<ContentBlock>,
    placed: Option<(usize, CacheControl)>,
) -> Vec<ContentBlock> {
    let Some((at, control)) = placed else {
        return blocks;
    };
    blocks
        .into_iter()
        .enumerate()
        .map(|(index, block)| match index == at {
            true => marked(block, control.clone()),
            false => block,
        })
        .collect()
}

fn raw_blocks_with(blocks: Vec<Value>, placed: Option<(usize, CacheControl)>) -> Vec<Value> {
    let Some((at, control)) = placed else {
        return blocks;
    };
    blocks
        .into_iter()
        .enumerate()
        .map(|(index, block)| match index == at {
            true => raw_marked(block, &control),
            false => block,
        })
        .collect()
}

fn raw_text_block(text: String) -> Value {
    Value::Object(Map::from_iter([
        ("type".to_owned(), Value::from("text")),
        ("text".to_owned(), Value::from(text)),
    ]))
}

fn message_with(
    message: Recognized<Message>,
    placed: Option<(usize, CacheControl)>,
) -> Recognized<Message> {
    match message {
        Recognized::Known(message) => Recognized::Known(Message {
            content: MessageContent::Blocks(blocks_with(
                match message.content {
                    MessageContent::Text(text) => vec![ContentBlock::text(text)],
                    MessageContent::Blocks(blocks) => blocks,
                },
                placed,
            )),
            ..message
        }),
        Recognized::Unrecognized(Value::Object(fields)) => Recognized::Unrecognized(Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| match value {
                    Value::String(text) if key == "content" => (
                        key,
                        Value::Array(raw_blocks_with(vec![raw_text_block(text)], placed.clone())),
                    ),
                    Value::Array(blocks) if key == "content" => {
                        (key, Value::Array(raw_blocks_with(blocks, placed.clone())))
                    }
                    value => (key, value),
                })
                .collect(),
        )),
        message => message,
    }
}

fn system_with(
    system: Recognized<SystemPrompt>,
    placed: Option<(usize, CacheControl)>,
) -> Recognized<SystemPrompt> {
    if placed.is_none() {
        return system;
    }
    match system {
        Recognized::Known(system) => Recognized::Known(SystemPrompt::Blocks(blocks_with(
            match system {
                SystemPrompt::Text(text) => vec![ContentBlock::text(text)],
                SystemPrompt::Blocks(blocks) => blocks,
            },
            placed,
        ))),
        Recognized::Unrecognized(Value::Array(blocks)) => {
            Recognized::Unrecognized(Value::Array(raw_blocks_with(blocks, placed)))
        }
        system => system,
    }
}

/// `request` with the breakpoints of `points` it has room for.
pub fn inject(request: MessagesRequest, points: &CachePoints) -> MessagesRequest {
    if points.is_empty() {
        return request;
    }
    let Plan { system, messages } = plan(points, &Census::of(&request));
    MessagesRequest {
        messages: request
            .messages
            .into_iter()
            .zip(messages)
            .map(|(message, placed)| message_with(message, placed))
            .collect(),
        params: MessagesOptionalParams {
            system: request
                .params
                .system
                .map(|prompt| system_with(prompt, system)),
            ..request.params
        },
        ..request
    }
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::formats::messages::MessagesRequest;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::{CachePoints, inject};

    fn points(written: Value) -> Option<CachePoints> {
        CachePoints::served(&serde_json::from_value(written).unwrap())
    }

    fn with_model(fields: Value) -> Value {
        let Value::Object(fields) = fields else {
            panic!("case fields are an object")
        };
        let mut body = json!({"model": "test-model", "max_tokens": 64});
        body.as_object_mut().unwrap().extend(fields);
        body
    }

    /// Each case is what Python's `maybe_inject_cache_control` sends for the same points and
    /// request, as `test_messages_points_match_the_rust_route` in
    /// `tests/unit/integrations/test_anthropic_cache_control_hook.py` holds it to.
    #[rstest]
    #[case::first_turn_text_system(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": "be terse", "messages": [{"role": "user", "content": "hi"}]}),
        json!({"system": [{"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}),
    )]
    #[case::continued_turn(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": "be terse", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]}),
        json!({"system": [{"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}, {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}, {"role": "user", "content": [{"type": "text", "text": "more", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}),
    )]
    #[case::no_system(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"messages": [{"role": "user", "content": "hi"}]}),
        json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}),
    )]
    #[case::block_system(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}),
        json!({"system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}),
    )]
    #[case::empty_block_system(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": [], "messages": [{"role": "user", "content": "hi"}]}),
        json!({"system": [], "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}),
    )]
    #[case::caller_marked_system(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "b"}], "messages": [{"role": "user", "content": "hi"}]}),
        json!({"system": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "b"}], "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}),
    )]
    #[case::caller_marked_first_block_of_last_message(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": "s", "messages": [{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "y"}]}]}),
        json!({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "y"}]}]}),
    )]
    #[case::null_mark_is_no_mark(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": "s", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": null}]}]}),
        json!({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}),
    )]
    #[case::four_caller_marks(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": "s", "messages": [{"role": "user", "content": [{"type": "text", "text": "1", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "2", "cache_control": {"type": "ephemeral"}}]}, {"role": "assistant", "content": [{"type": "text", "text": "3", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "4", "cache_control": {"type": "ephemeral"}}]}, {"role": "user", "content": "q"}]}),
        json!({"system": "s", "messages": [{"role": "user", "content": [{"type": "text", "text": "1", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "2", "cache_control": {"type": "ephemeral"}}]}, {"role": "assistant", "content": [{"type": "text", "text": "3", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "4", "cache_control": {"type": "ephemeral"}}]}, {"role": "user", "content": [{"type": "text", "text": "q"}]}]}),
    )]
    #[case::three_caller_marks_leave_one_slot(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": "s", "messages": [{"role": "user", "content": [{"type": "text", "text": "1", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "2", "cache_control": {"type": "ephemeral"}}]}, {"role": "assistant", "content": [{"type": "text", "text": "3", "cache_control": {"type": "ephemeral"}}]}, {"role": "user", "content": "q"}]}),
        json!({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "1", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "2", "cache_control": {"type": "ephemeral"}}]}, {"role": "assistant", "content": [{"type": "text", "text": "3", "cache_control": {"type": "ephemeral"}}]}, {"role": "user", "content": [{"type": "text", "text": "q"}]}]}),
    )]
    #[case::three_marked_tools_leave_one_slot(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": "s", "tools": [{"name": "a", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}, {"name": "b", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}, {"name": "c", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}], "messages": [{"role": "user", "content": "q"}]}),
        json!({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "tools": [{"name": "a", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}, {"name": "b", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}, {"name": "c", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}]}),
    )]
    #[case::tool_openai_shape_mark(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": "s", "tools": [{"type": "function", "function": {"name": "a", "parameters": {}, "cache_control": {"type": "ephemeral"}}}, {"type": "function", "function": {"name": "b", "parameters": {}, "cache_control": {"type": "ephemeral"}}}, {"type": "function", "function": {"name": "c", "parameters": {}, "cache_control": {"type": "ephemeral"}}}], "messages": [{"role": "user", "content": "q"}]}),
        json!({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "tools": [{"type": "function", "function": {"name": "a", "parameters": {}, "cache_control": {"type": "ephemeral"}}}, {"type": "function", "function": {"name": "b", "parameters": {}, "cache_control": {"type": "ephemeral"}}}, {"type": "function", "function": {"name": "c", "parameters": {}, "cache_control": {"type": "ephemeral"}}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}]}),
    )]
    #[case::automatic_caching_takes_a_slot(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}, {"location": "message", "role": "user"}]),
        json!({"system": "s", "cache_control": {"type": "ephemeral"}, "messages": [{"role": "user", "content": "1"}, {"role": "assistant", "content": "2"}, {"role": "user", "content": "3"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "5"}]}),
        json!({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "cache_control": {"type": "ephemeral"}, "messages": [{"role": "user", "content": [{"type": "text", "text": "1", "cache_control": {"type": "ephemeral"}}]}, {"role": "assistant", "content": [{"type": "text", "text": "2"}]}, {"role": "user", "content": [{"type": "text", "text": "3"}]}, {"role": "assistant", "content": [{"type": "text", "text": "4"}]}, {"role": "user", "content": [{"type": "text", "text": "5", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}),
    )]
    #[case::thinking_is_never_marked(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "text", "text": "answer"}, {"type": "thinking", "thinking": "hm", "signature": "s"}]}]}),
        json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}, {"role": "assistant", "content": [{"type": "text", "text": "answer", "cache_control": {"type": "ephemeral", "ttl": "5m"}}, {"type": "thinking", "thinking": "hm", "signature": "s"}]}]}),
    )]
    #[case::only_thinking_is_left_unmarked(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]}]}),
        json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}, {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]}]}),
    )]
    #[case::tool_result_is_marked(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]}),
        json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}, {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}),
    )]
    #[case::empty_content_list(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"messages": [{"role": "user", "content": []}]}),
        json!({"messages": [{"role": "user", "content": []}]}),
    )]
    #[case::role_points_stop_at_the_limit(
        json!([{"location": "message", "role": "user"}]),
        json!({"messages": [{"role": "user", "content": "1"}, {"role": "assistant", "content": "2"}, {"role": "user", "content": "3"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "5"}, {"role": "assistant", "content": "6"}, {"role": "user", "content": "7"}, {"role": "assistant", "content": "8"}, {"role": "user", "content": "9"}]}),
        json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "1", "cache_control": {"type": "ephemeral"}}]}, {"role": "assistant", "content": [{"type": "text", "text": "2"}]}, {"role": "user", "content": [{"type": "text", "text": "3", "cache_control": {"type": "ephemeral"}}]}, {"role": "assistant", "content": [{"type": "text", "text": "4"}]}, {"role": "user", "content": [{"type": "text", "text": "5", "cache_control": {"type": "ephemeral"}}]}, {"role": "assistant", "content": [{"type": "text", "text": "6"}]}, {"role": "user", "content": [{"type": "text", "text": "7", "cache_control": {"type": "ephemeral"}}]}, {"role": "assistant", "content": [{"type": "text", "text": "8"}]}, {"role": "user", "content": [{"type": "text", "text": "9"}]}]}),
    )]
    #[case::index_wins_over_role(
        json!([{"location": "message", "index": 1, "role": "user"}]),
        json!({"messages": [{"role": "user", "content": "1"}, {"role": "assistant", "content": "2"}, {"role": "user", "content": "3"}]}),
        json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "1"}]}, {"role": "assistant", "content": [{"type": "text", "text": "2", "cache_control": {"type": "ephemeral"}}]}, {"role": "user", "content": [{"type": "text", "text": "3"}]}]}),
    )]
    #[case::index_out_of_range(
        json!([{"location": "message", "index": 5}, {"location": "message", "index": -4}]),
        json!({"messages": [{"role": "user", "content": "1"}, {"role": "assistant", "content": "2"}, {"role": "user", "content": "3"}]}),
        json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "1"}]}, {"role": "assistant", "content": [{"type": "text", "text": "2"}]}, {"role": "user", "content": [{"type": "text", "text": "3"}]}]}),
    )]
    #[case::system_role_ignores_its_index(
        json!([{"location": "message", "role": "system", "index": 0}]),
        json!({"system": "s", "messages": [{"role": "user", "content": "1"}]}),
        json!({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "1"}]}]}),
    )]
    #[case::first_system_point_wins(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "5m"}}, {"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}]),
        json!({"system": "s", "messages": [{"role": "user", "content": "1"}]}),
        json!({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "5m"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "1"}]}]}),
    )]
    #[case::no_control_is_ephemeral(
        json!([{"location": "message", "role": "system"}, {"location": "message", "index": -1, "control": {}}]),
        json!({"system": "s", "messages": [{"role": "user", "content": "1"}]}),
        json!({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "1", "cache_control": {"type": "ephemeral"}}]}]}),
    )]
    #[case::earlier_point_wins_a_message(
        json!([{"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"messages": [{"role": "user", "content": "1"}]}),
        json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "1", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]}),
    )]
    #[case::message_level_mark_counts(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": "s", "messages": [{"role": "user", "content": "hi", "cache_control": {"type": "ephemeral"}}]}),
        json!({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}], "cache_control": {"type": "ephemeral"}}]}),
    )]
    #[case::unparsed_message(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": "s", "messages": [{"role": "user", "content": "1"}, {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": "x"}, {"type": "text", "text": "b"}]}, {"role": "user", "content": [{"type": "text", "text": 5}, {"type": "text", "text": "last"}]}]}),
        json!({"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "1"}]}, {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": "x"}, {"type": "text", "text": "b"}]}, {"role": "user", "content": [{"type": "text", "text": 5}, {"type": "text", "text": "last", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}),
    )]
    #[case::unparsed_system(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": [{"type": "text", "text": 5}, {"type": "text", "text": "b"}], "messages": [{"role": "user", "content": "1"}]}),
        json!({"system": [{"type": "text", "text": 5}, {"type": "text", "text": "b", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "1", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}),
    )]
    #[case::empty_text_system(
        json!([{"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}, {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]),
        json!({"system": "", "messages": [{"role": "user", "content": "1"}]}),
        json!({"system": [{"type": "text", "text": "", "cache_control": {"type": "ephemeral", "ttl": "1h"}}], "messages": [{"role": "user", "content": [{"type": "text", "text": "1", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}),
    )]
    fn a_request_is_sent_with_the_breakpoints_python_places(
        #[case] written: Value,
        #[case] request: Value,
        #[case] sent: Value,
    ) {
        let request: MessagesRequest = serde_json::from_value(with_model(request)).unwrap();

        let injected = inject(request, &points(written).unwrap());

        assert_eq!(serde_json::to_value(injected).unwrap(), with_model(sent));
    }

    #[rstest]
    fn no_points_leave_the_request_as_sent() {
        let sent =
            with_model(json!({"system": "s", "messages": [{"role": "user", "content": "hi"}]}));
        let request: MessagesRequest = serde_json::from_value(sent.clone()).unwrap();

        let injected = inject(request, &points(json!([])).unwrap());

        assert_eq!(serde_json::to_value(injected).unwrap(), sent);
    }

    #[rstest]
    #[case::a_tool_config_point(json!([{"location": "message", "role": "system"}, {"location": "tool_config"}]))]
    #[case::a_point_that_names_no_target(json!([{"location": "message"}]))]
    #[case::an_index_written_as_text(json!([{"location": "message", "index": "-1"}]))]
    #[case::an_entry_that_is_no_point(json!(["message"]))]
    #[case::a_value_that_is_no_list(json!({"location": "message", "role": "system"}))]
    fn points_the_route_does_not_place_are_not_served(#[case] written: Value) {
        assert_eq!(points(written), None);
    }
}
