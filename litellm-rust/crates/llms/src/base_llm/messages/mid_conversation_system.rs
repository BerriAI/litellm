use litellm_llms_types::formats::messages::{
    ContentBlock, ContentBlockType, Message, MessageContent, SystemPrompt,
};
use serde_json::Map;

/// Python's `CONVERTED_SYSTEM_NOTE`.
pub const CONVERTED_SYSTEM_NOTE: &str = "Operator note (not from the user): the following was originally a mid-conversation system-role reminder.";

const SYSTEM_ROLE: &str = "system";
const USER_ROLE: &str = "user";

pub fn as_system_content_blocks(value: Option<SystemPrompt>) -> Vec<ContentBlock> {
    match value {
        None => Vec::new(),
        Some(SystemPrompt::Text(text)) => vec![ContentBlock::text(text)],
        Some(SystemPrompt::Blocks(blocks)) => blocks,
    }
}

pub fn message_content_blocks(content: MessageContent) -> Vec<ContentBlock> {
    match content {
        MessageContent::Text(text) => vec![ContentBlock::text(text)],
        MessageContent::Blocks(blocks) => blocks,
    }
}

pub fn is_system_role_message(message: &Message) -> bool {
    message.role == SYSTEM_ROLE
}

pub fn system_role_message_as_user(message: Message) -> Message {
    Message {
        role: USER_ROLE.into(),
        content: MessageContent::Blocks(
            [ContentBlock::text(CONVERTED_SYSTEM_NOTE)]
                .into_iter()
                .chain(message_content_blocks(message.content))
                .collect(),
        ),
        extra: Map::new(),
    }
}

pub fn opens_with_tool_results(message: &Message) -> bool {
    message.role == USER_ROLE
        && matches!(
            &message.content,
            MessageContent::Blocks(blocks)
                if blocks.first().and_then(|block| block.block_type.as_ref()) == Some(&ContentBlockType::ToolResult)
        )
}

fn system_run_placed_after_tool_results(
    system_run: Vec<Message>,
    follower_run: Vec<Message>,
) -> Vec<Message> {
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

fn runs(messages: Vec<Message>) -> Vec<Vec<Message>> {
    messages
        .into_iter()
        .fold(Vec::new(), |mut runs: Vec<Vec<Message>>, message| {
            match runs.last_mut() {
                Some(run)
                    if is_system_role_message(&run[0]) == is_system_role_message(&message) =>
                {
                    run.push(message);
                }
                _ => runs.push(vec![message]),
            }
            runs
        })
}

fn system_turns_after_tool_results(messages: Vec<Message>) -> Vec<Message> {
    let mut runs = runs(messages).into_iter();
    let mut placed = Vec::new();
    while let Some(run) = runs.next() {
        if !is_system_role_message(&run[0]) {
            placed.extend(run);
            continue;
        }
        let follower_run = runs.next().unwrap_or_default();
        placed.extend(system_run_placed_after_tool_results(run, follower_run));
    }
    placed
}

pub fn convert_mid_conversation_system_turns(messages: Vec<Message>) -> Vec<Message> {
    system_turns_after_tool_results(messages)
        .into_iter()
        .map(|message| {
            if is_system_role_message(&message) {
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

    fn messages(value: Value) -> Vec<Message> {
        serde_json::from_value(value).unwrap()
    }

    fn roles(messages: &[Message]) -> Vec<&str> {
        messages
            .iter()
            .map(|message| message.role.as_str())
            .collect()
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
        assert!(
            converted
                .iter()
                .all(|message| !is_system_role_message(message))
        );
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
