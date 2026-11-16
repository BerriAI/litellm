use litellm_llms_types::{
    formats::messages::{ContentBlock, ContentBlockType, Message, MessageContent, SystemPrompt},
    recognized::Recognized,
};
use serde_json::{Map, Value};

/// Python's `CONVERTED_SYSTEM_NOTE`.
pub const CONVERTED_SYSTEM_NOTE: &str = "Operator note (not from the user): the following was originally a mid-conversation system-role reminder.";

const SYSTEM_ROLE: &str = "system";
const USER_ROLE: &str = "user";
const CONTENT: &str = "content";
const ROLE: &str = "role";

/// Python's `as_system_content_blocks` over a system prompt. One Rust could not parse keeps
/// its items as sent: a list gives its items, any other value is one item.
pub fn as_system_content_blocks(
    value: Option<Recognized<SystemPrompt>>,
) -> Vec<Recognized<ContentBlock>> {
    match value {
        None => Vec::new(),
        Some(Recognized::Known(SystemPrompt::Text(text))) => {
            vec![Recognized::Known(ContentBlock::text(text))]
        }
        Some(Recognized::Known(SystemPrompt::Blocks(blocks))) => {
            blocks.into_iter().map(Recognized::Known).collect()
        }
        Some(Recognized::Unrecognized(value)) => raw_content_blocks(Some(value)),
    }
}

/// `as_system_content_blocks` over content Rust could not parse: missing or `null` content
/// gives nothing, a string one text block, a list its items as sent and any other value itself.
fn raw_content_blocks(content: Option<Value>) -> Vec<Recognized<ContentBlock>> {
    match content {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(text)) => vec![Recognized::Known(ContentBlock::text(text))],
        Some(Value::Array(items)) => items.into_iter().map(Recognized::Unrecognized).collect(),
        Some(other) => vec![Recognized::Unrecognized(other)],
    }
}

/// The content of a system turn as system blocks, whether or not the message parsed.
pub fn message_content_blocks(message: Recognized<Message>) -> Vec<Recognized<ContentBlock>> {
    match message {
        Recognized::Known(message) => match message.content {
            MessageContent::Text(text) => vec![Recognized::Known(ContentBlock::text(text))],
            MessageContent::Blocks(blocks) => blocks.into_iter().map(Recognized::Known).collect(),
        },
        Recognized::Unrecognized(Value::Object(mut fields)) => {
            raw_content_blocks(fields.shift_remove(CONTENT))
        }
        Recognized::Unrecognized(_) => Vec::new(),
    }
}

/// The blocks as a system prompt: typed when every block parsed, else the list as sent.
pub fn system_prompt(blocks: Vec<Recognized<ContentBlock>>) -> Recognized<SystemPrompt> {
    match known_blocks(blocks) {
        Ok(blocks) => Recognized::Known(SystemPrompt::Blocks(blocks)),
        Err(raw) => Recognized::Unrecognized(Value::Array(raw)),
    }
}

/// Every block typed, or every block as JSON when one of them did not parse.
fn known_blocks(blocks: Vec<Recognized<ContentBlock>>) -> Result<Vec<ContentBlock>, Vec<Value>> {
    if blocks.iter().all(|block| block.known().is_some()) {
        return Ok(blocks
            .into_iter()
            .filter_map(|block| match block {
                Recognized::Known(block) => Some(block),
                Recognized::Unrecognized(_) => None,
            })
            .collect());
    }
    Err(blocks
        .into_iter()
        .map(|block| match block {
            Recognized::Known(block) => {
                serde_json::to_value(block).expect("a content block serializes")
            }
            Recognized::Unrecognized(block) => block,
        })
        .collect())
}

pub fn is_system_role_message(message: &Message) -> bool {
    message.role == SYSTEM_ROLE
}

/// Python's `is_system_role_message` reads only the role, so a system message Rust could not
/// parse, such as one sent without content, is still a system turn.
pub fn is_system_turn(message: &Recognized<Message>) -> bool {
    message_role(message) == Some(SYSTEM_ROLE)
}

/// The `role` of a message whether or not it parsed, as Python reads it off any mapping.
pub fn message_role(message: &Recognized<Message>) -> Option<&str> {
    match message {
        Recognized::Known(message) => Some(&message.role),
        Recognized::Unrecognized(Value::Object(fields)) => fields.get(ROLE).and_then(Value::as_str),
        Recognized::Unrecognized(_) => None,
    }
}

/// Python's `system_role_message_as_user`: the note, then the turn's content as system blocks.
/// The turn stays unparsed when one of its blocks did not parse.
pub fn system_role_message_as_user(message: Recognized<Message>) -> Recognized<Message> {
    let blocks: Vec<Recognized<ContentBlock>> =
        [Recognized::Known(ContentBlock::text(CONVERTED_SYSTEM_NOTE))]
            .into_iter()
            .chain(message_content_blocks(message))
            .collect();
    match known_blocks(blocks) {
        Ok(blocks) => Recognized::Known(Message {
            role: USER_ROLE.into(),
            content: MessageContent::Blocks(blocks),
            extra: Map::new(),
        }),
        Err(raw) => Recognized::Unrecognized(Value::Object(Map::from_iter([
            (ROLE.to_owned(), Value::String(USER_ROLE.into())),
            (CONTENT.to_owned(), Value::Array(raw)),
        ]))),
    }
}

pub fn opens_with_tool_results(message: &Recognized<Message>) -> bool {
    match message {
        Recognized::Known(message) => {
            message.role == USER_ROLE
                && matches!(
                    &message.content,
                    MessageContent::Blocks(blocks)
                        if blocks.first().and_then(|block| block.block_type.as_ref()) == Some(&ContentBlockType::ToolResult)
                )
        }
        Recognized::Unrecognized(message) => {
            message.get(ROLE).and_then(Value::as_str) == Some(USER_ROLE)
                && message
                    .get(CONTENT)
                    .and_then(Value::as_array)
                    .and_then(|blocks| blocks.first())
                    .and_then(|block| block.get("type"))
                    .and_then(Value::as_str)
                    == Some("tool_result")
        }
    }
}

fn system_run_placed_after_tool_results(
    system_run: Vec<Recognized<Message>>,
    follower_run: Vec<Recognized<Message>>,
) -> Vec<Recognized<Message>> {
    let mut follower_run = follower_run.into_iter();
    match follower_run.next() {
        Some(first) if opens_with_tool_results(&first) => [first]
            .into_iter()
            .chain(system_run)
            .chain(follower_run)
            .collect(),
        first => system_run
            .into_iter()
            .chain(first)
            .chain(follower_run)
            .collect(),
    }
}

fn runs(messages: Vec<Recognized<Message>>) -> Vec<Vec<Recognized<Message>>> {
    messages.into_iter().fold(
        Vec::new(),
        |mut runs: Vec<Vec<Recognized<Message>>>, message| {
            match runs.last_mut() {
                Some(run) if is_system_turn(&run[0]) == is_system_turn(&message) => {
                    run.push(message);
                }
                _ => runs.push(vec![message]),
            }
            runs
        },
    )
}

fn system_turns_after_tool_results(messages: Vec<Recognized<Message>>) -> Vec<Recognized<Message>> {
    let mut runs = runs(messages).into_iter();
    let mut placed = Vec::new();
    while let Some(run) = runs.next() {
        if !is_system_turn(&run[0]) {
            placed.extend(run);
            continue;
        }
        let follower_run = runs.next().unwrap_or_default();
        placed.extend(system_run_placed_after_tool_results(run, follower_run));
    }
    placed
}

/// Python's `convert_mid_conversation_system_turns`, over every message whether or not it
/// parsed, as Python reads the role off any mapping.
pub fn convert_mid_conversation_system_turns(
    messages: Vec<Recognized<Message>>,
) -> Vec<Recognized<Message>> {
    system_turns_after_tool_results(messages)
        .into_iter()
        .map(|message| {
            if is_system_turn(&message) {
                system_role_message_as_user(message)
            } else {
                message
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn messages(value: Value) -> Vec<Recognized<Message>> {
        serde_json::from_value(value).unwrap()
    }

    fn roles(messages: &[Recognized<Message>]) -> Vec<&str> {
        messages.iter().filter_map(message_role).collect()
    }

    #[rstest]
    #[case::no_system_turns(
        json!([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]),
        vec!["user", "assistant"],
    )]
    #[case::a_turn_between_messages_becomes_a_user_turn_in_place(
        json!([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "system", "content": "reminder"},
            {"role": "user", "content": "c"}
        ]),
        vec!["user", "assistant", "user", "user"],
    )]
    #[case::a_run_between_a_tool_use_and_its_result_moves_after_the_result(
        json!([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "n", "input": {}}]},
            {"role": "system", "content": "reminder"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}, {"type": "text", "text": "next"}]},
            {"role": "user", "content": "d"}
        ]),
        vec!["user", "assistant", "user", "user", "user"],
    )]
    fn turns_are_converted_in_place_and_never_split_a_tool_call_from_its_result(
        #[case] input: Value,
        #[case] expected_roles: Vec<&str>,
    ) {
        let converted = convert_mid_conversation_system_turns(messages(input));

        assert_eq!(roles(&converted), expected_roles);
        assert!(converted.iter().all(|message| !is_system_turn(message)));
    }

    #[rstest]
    fn a_converted_turn_carries_the_note_then_the_original_content_only() {
        let converted = convert_mid_conversation_system_turns(messages(json!([
            {"role": "user", "content": "a"},
            {"role": "system", "content": "reminder", "name": "ignored"}
        ])));

        assert_eq!(
            serde_json::to_value(&converted[1]).unwrap(),
            json!({"role": "user", "content": [
                {"type": "text", "text": CONVERTED_SYSTEM_NOTE},
                {"type": "text", "text": "reminder"}
            ]})
        );
    }

    #[rstest]
    fn the_tool_result_turn_keeps_its_place_before_the_moved_run() {
        let converted = convert_mid_conversation_system_turns(messages(json!([
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "n", "input": {}}]},
            {"role": "system", "content": "reminder"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}
        ])));

        assert!(opens_with_tool_results(&converted[1]));
        assert_eq!(
            serde_json::to_value(&converted[2]).unwrap()["content"][1],
            json!({"type": "text", "text": "reminder"})
        );
    }
}
