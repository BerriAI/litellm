//! Provider-neutral conversation shape.
//!
//! Both Anthropic Messages and Bedrock Converse want the same thing out of an
//! OpenAI message list: the system prompt lifted out, consecutive same-role
//! turns merged, and text blocks that are never empty. That normalization is
//! shared here so a provider config only renders the result into its own wire
//! shape.
//!
//! Mirrors Python's `anthropic_messages_pt` /
//! `_bedrock_converse_messages_pt` for the text-only surface this route
//! accepts; anything richer is declined upstream by the capability gate.

use crate::constants::EMPTY_TEXT_PLACEHOLDER;

use super::types::{ChatMessage, ChatMessageContent};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnRole {
    User,
    Assistant,
}

impl TurnRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Turn {
    pub role: TurnRole,
    pub texts: Vec<String>,
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
            let texts = message
                .content
                .as_ref()
                .map(message_texts)
                .unwrap_or_default()
                .into_iter()
                .map(sanitize);
            match turns.last_mut() {
                Some(last) if last.role == role => last.texts.extend(texts),
                _ => turns.push(Turn {
                    role,
                    texts: texts.collect(),
                }),
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
                texts: last
                    .texts
                    .iter()
                    .map(|text| text.trim_end().to_string())
                    .collect(),
            }])
            .collect(),
        _ => turns,
    };

    Conversation { system, turns }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn messages(value: serde_json::Value) -> Vec<ChatMessage> {
        serde_json::from_value(value).expect("valid messages")
    }

    #[test]
    fn lifts_system_messages_out_of_the_turn_list() {
        let conversation = build_conversation(&messages(json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"}
        ])));
        assert_eq!(conversation.system, vec!["be terse".to_string()]);
        assert_eq!(
            conversation.turns,
            vec![Turn {
                role: TurnRole::User,
                texts: vec!["hi".to_string()]
            }]
        );
    }

    #[test]
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
                Turn {
                    role: TurnRole::User,
                    texts: vec!["one".to_string(), "two".to_string()]
                },
                Turn {
                    role: TurnRole::Assistant,
                    texts: vec!["ack".to_string()]
                },
                Turn {
                    role: TurnRole::User,
                    texts: vec!["three".to_string()]
                },
            ]
        );
    }

    #[test]
    fn flattens_text_parts_in_order() {
        let conversation = build_conversation(&messages(json!([
            {"role": "user", "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"}
            ]}
        ])));
        assert_eq!(
            conversation.turns[0].texts,
            vec!["first".to_string(), "second".to_string()]
        );
    }

    #[test]
    fn rewrites_empty_and_whitespace_only_text_to_the_python_placeholder() {
        let conversation = build_conversation(&messages(json!([
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "   "},
            {"role": "user", "content": "real"}
        ])));
        assert_eq!(conversation.turns[0].texts, vec![EMPTY_TEXT_PLACEHOLDER]);
        assert_eq!(conversation.turns[1].texts, vec![EMPTY_TEXT_PLACEHOLDER]);
    }

    #[test]
    fn right_strips_only_the_final_assistant_turn() {
        let conversation = build_conversation(&messages(json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "kept  "},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "stripped  "}
        ])));
        assert_eq!(conversation.turns[1].texts, vec!["kept  ".to_string()]);
        assert_eq!(conversation.turns[3].texts, vec!["stripped".to_string()]);
    }

    #[test]
    fn does_not_strip_when_the_last_turn_is_a_user_turn() {
        let conversation = build_conversation(&messages(json!([
            {"role": "assistant", "content": "kept  "},
            {"role": "user", "content": "hi  "}
        ])));
        assert_eq!(conversation.turns[0].texts, vec!["kept  ".to_string()]);
        assert_eq!(conversation.turns[1].texts, vec!["hi  ".to_string()]);
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
}
