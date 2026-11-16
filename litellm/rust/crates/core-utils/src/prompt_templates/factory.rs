//! Provider-neutral conversation shape.
//!
//! Both Anthropic Messages and Bedrock Converse want the same thing out of an
//! OpenAI message list: the system prompt lifted out, consecutive same-role
//! turns merged, and text blocks that are never empty. That normalization is
//! shared here so a provider config only renders the result into its own wire
//! shape.
//!
//! Mirrors Python's `anthropic_messages_pt` /
//! `_bedrock_converse_messages_pt` for the text surface this route accepts,
//! plus assistant `thinking_blocks` carried raw for a provider to replay;
//! anything richer is declined upstream by the capability gate.

use litellm_llms_types::formats::chat_completions::{ChatMessage, ChatMessageContent};
use serde_json::Value;
use strum::IntoStaticStr;

pub const THINKING_BLOCKS_FIELD: &str = "thinking_blocks";

pub const EMPTY_TEXT_PLACEHOLDER: &str =
    "[System: Empty message content sanitised to satisfy protocol]";

#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq)]
pub enum TurnRole {
    #[strum(serialize = "user")]
    User,
    #[strum(serialize = "assistant")]
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TurnBlock {
    Text(String),
    Thinking(Value),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Turn {
    pub role: TurnRole,
    pub blocks: Vec<TurnBlock>,
}

impl Turn {
    pub fn texts(&self) -> impl Iterator<Item = &str> {
        self.blocks.iter().filter_map(|block| match block {
            TurnBlock::Text(text) => Some(text.as_str()),
            TurnBlock::Thinking(_) => None,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Conversation {
    pub system: Vec<String>,
    pub turns: Vec<Turn>,
}

/// True when the conversation can be sent as-is.
///
/// Python inserts a placeholder first user turn only under
/// `litellm.modify_params`, which the core cannot see, so a conversation that
/// does not open on a user turn is declined rather than guessed at.
impl Conversation {
    pub fn opens_on_user_turn(&self) -> bool {
        self.turns
            .first()
            .is_some_and(|turn| turn.role == TurnRole::User)
    }
}

fn message_texts(content: &ChatMessageContent) -> Vec<String> {
    match content {
        ChatMessageContent::Text(text) => vec![text.clone()],
        ChatMessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(|text| text.as_str()))
            .map(str::to_string)
            .collect(),
    }
}

/// Python rewrites empty or whitespace-only text rather than dropping it, so an
/// entirely empty content list never reaches a provider that rejects one.
fn sanitize(text: String) -> String {
    if text.trim().is_empty() {
        return EMPTY_TEXT_PLACEHOLDER.to_string();
    }
    text
}

/// Python adds an assistant message's thinking blocks ahead of its text, one
/// message at a time, so merged turns keep each message's own order.
fn message_blocks(message: &ChatMessage, role: TurnRole) -> Vec<TurnBlock> {
    let thinking = match role {
        TurnRole::Assistant => message
            .extra
            .get(THINKING_BLOCKS_FIELD)
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        TurnRole::User => Vec::new(),
    };
    let texts = message
        .content
        .as_ref()
        .map(message_texts)
        .unwrap_or_default();
    thinking
        .into_iter()
        .map(TurnBlock::Thinking)
        .chain(texts.into_iter().map(sanitize).map(TurnBlock::Text))
        .collect()
}

pub fn build_conversation(messages: &[ChatMessage]) -> Conversation {
    let system = messages
        .iter()
        .filter(|message| message.role == "system")
        .filter_map(|message| message.content.as_ref())
        .flat_map(message_texts)
        .filter(|text| !text.is_empty())
        .collect();

    let turns = messages
        .iter()
        .filter(|message| message.role != "system")
        .fold(Vec::<Turn>::new(), |mut turns, message| {
            let role = if message.role == "assistant" {
                TurnRole::Assistant
            } else {
                TurnRole::User
            };
            let blocks = message_blocks(message, role);
            match turns.last_mut() {
                Some(last) if last.role == role => last.blocks.extend(blocks),
                _ => turns.push(Turn { role, blocks }),
            }
            turns
        });

    // Anthropic and Bedrock both reject trailing whitespace on the final
    // assistant turn, so Python right-strips it there; mirror that exactly.
    let turns = match turns.split_last() {
        Some((last, rest)) if last.role == TurnRole::Assistant => rest
            .iter()
            .cloned()
            .chain([Turn {
                role: last.role,
                blocks: last
                    .blocks
                    .iter()
                    .map(|block| match block {
                        TurnBlock::Text(text) => TurnBlock::Text(text.trim_end().to_string()),
                        TurnBlock::Thinking(thinking) => TurnBlock::Thinking(thinking.clone()),
                    })
                    .collect(),
            }])
            .collect(),
        _ => turns,
    };

    Conversation { system, turns }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn messages(value: serde_json::Value) -> Vec<ChatMessage> {
        serde_json::from_value(value).expect("valid messages")
    }

    fn text_turn(role: TurnRole, texts: &[&str]) -> Turn {
        Turn {
            role,
            blocks: texts
                .iter()
                .map(|text| TurnBlock::Text(text.to_string()))
                .collect(),
        }
    }

    fn texts(turn: &Turn) -> Vec<&str> {
        turn.texts().collect()
    }

    #[rstest]
    fn lifts_system_messages_out_of_the_turn_list() {
        let conversation = build_conversation(&messages(json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"}
        ])));
        assert_eq!(conversation.system, vec!["be terse".to_string()]);
        assert_eq!(conversation.turns, vec![text_turn(TurnRole::User, &["hi"])]);
    }

    #[rstest]
    fn merges_consecutive_same_role_turns() {
        let conversation = build_conversation(&messages(json!([
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "three"}
        ])));
        assert_eq!(
            conversation.turns,
            vec![
                text_turn(TurnRole::User, &["one", "two"]),
                text_turn(TurnRole::Assistant, &["ack"]),
                text_turn(TurnRole::User, &["three"]),
            ]
        );
    }

    #[rstest]
    fn flattens_text_parts_in_order() {
        let conversation = build_conversation(&messages(json!([
            {"role": "user", "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"}
            ]}
        ])));
        assert_eq!(texts(&conversation.turns[0]), vec!["first", "second"]);
    }

    #[rstest]
    fn rewrites_empty_and_whitespace_only_text_to_the_python_placeholder() {
        let conversation = build_conversation(&messages(json!([
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "   "},
            {"role": "user", "content": "real"}
        ])));
        // Must equal `_EMPTY_TEXT_PLACEHOLDER` in litellm/litellm_core_utils/prompt_templates/factory.py
        let placeholder = "[System: Empty message content sanitised to satisfy protocol]";
        assert_eq!(texts(&conversation.turns[0]), vec![placeholder]);
        assert_eq!(texts(&conversation.turns[1]), vec![placeholder]);
    }

    #[rstest]
    fn right_strips_only_the_final_assistant_turn() {
        let conversation = build_conversation(&messages(json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "kept  "},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "stripped  "}
        ])));
        assert_eq!(texts(&conversation.turns[1]), vec!["kept  "]);
        assert_eq!(texts(&conversation.turns[3]), vec!["stripped"]);
    }

    #[rstest]
    fn does_not_strip_when_the_last_turn_is_a_user_turn() {
        let conversation = build_conversation(&messages(json!([
            {"role": "assistant", "content": "kept  "},
            {"role": "user", "content": "hi  "}
        ])));
        assert_eq!(texts(&conversation.turns[0]), vec!["kept  "]);
        assert_eq!(texts(&conversation.turns[1]), vec!["hi  "]);
    }

    #[test]
    fn reports_whether_the_conversation_opens_on_a_user_turn() {
        assert!(
            build_conversation(&messages(json!([{"role": "user", "content": "hi"}])))
                .opens_on_user_turn()
        );
        assert!(
            !build_conversation(&messages(json!([{"role": "assistant", "content": "hi"}])))
                .opens_on_user_turn()
        );
        assert!(!Conversation::default().opens_on_user_turn());
    }

    #[test]
    fn drops_empty_system_text_the_way_python_skips_empty_system_blocks() {
        let conversation = build_conversation(&messages(json!([
            {"role": "system", "content": ""},
            {"role": "system", "content": "kept"},
            {"role": "user", "content": "hi"}
        ])));
        assert_eq!(conversation.system, vec!["kept".to_string()]);
    }

    #[rstest]
    fn puts_each_assistant_messages_thinking_blocks_ahead_of_its_own_text() {
        let first = json!({"type": "thinking", "thinking": "a", "signature": "s1"});
        let second = json!({"type": "redacted_thinking", "data": "d2"});
        let conversation = build_conversation(&messages(json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "one", "thinking_blocks": [first]},
            {"role": "assistant", "content": "two", "thinking_blocks": [second]},
            {"role": "user", "content": "more"}
        ])));
        assert_eq!(
            conversation.turns[1].blocks,
            vec![
                TurnBlock::Thinking(first),
                TurnBlock::Text("one".to_string()),
                TurnBlock::Thinking(second),
                TurnBlock::Text("two".to_string()),
            ]
        );
    }
}
