use serde::{Deserialize, Serialize};

use crate::claude_membership;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Facts {
    pub route: String,
    pub request_present: bool,
    pub router_present: Option<bool>,
    pub normalized: Option<bool>,
    pub model_loaded: bool,
    pub requested: Option<String>,
    pub body_loaded: bool,
    pub body_model: Option<String>,
    pub settings_loaded: bool,
    pub membership: Vec<claude_membership::Facts>,
    pub marked: bool,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Complete,
    ReadRouter,
    ReadNormalized,
    ReadModel,
    ReadBodyModel,
    ReadSettings,
    ReadMembership { name: String, index: usize },
    MarkNormalized,
    Rewrite { target: String },
}

fn unmarked(name: &str) -> &str {
    if name.to_ascii_lowercase().ends_with("[1m]") {
        &name[..name.len() - 4]
    } else {
        name
    }
}

fn decoded(name: &str) -> Option<String> {
    let encoded = unmarked(name).strip_prefix("claude-router-")?;
    if !encoded.len().is_multiple_of(2)
        || !encoded
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return None;
    }
    let bytes: Option<Vec<u8>> = encoded
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok())
        .collect();
    String::from_utf8(bytes?)
        .ok()
        .filter(|value| !value.is_empty())
}

fn membership(facts: &Facts, index: usize, name: &str) -> Result<bool, Decision> {
    if let Some(read) = facts.membership.get(index).filter(|read| read.name == name)
        && let claude_membership::Decision::Complete { present } = claude_membership::decide(read)
    {
        return Ok(present);
    }
    Err(Decision::ReadMembership {
        name: name.into(),
        index,
    })
}

fn target(facts: &Facts, name: &str) -> Result<Option<String>, Decision> {
    if membership(facts, 0, name)? {
        return Ok(None);
    }
    let plain = unmarked(name);
    let marked = plain != name;
    if marked && membership(facts, 1, plain)? {
        return Ok(Some(plain.into()));
    }
    let Some(model) = decoded(name) else {
        return Ok(None);
    };
    Ok(membership(facts, if marked { 2 } else { 1 }, &model)?.then_some(model))
}

pub fn decide(facts: &Facts) -> Decision {
    if !matches!(
        facts.route.as_str(),
        "/messages"
            | "/v1/messages"
            | "/chat/completions"
            | "/v1/chat/completions"
            | "/responses"
            | "/v1/responses"
    ) {
        return Decision::Complete;
    }
    match facts.router_present {
        None => return Decision::ReadRouter,
        Some(false) => return Decision::Complete,
        Some(true) => (),
    }
    if facts.request_present {
        match facts.normalized {
            None => return Decision::ReadNormalized,
            Some(true) => return Decision::Complete,
            Some(false) => (),
        }
    }
    if !facts.model_loaded {
        return Decision::ReadModel;
    }
    let Some(requested) = facts.requested.as_deref() else {
        return Decision::Complete;
    };
    if !facts.body_loaded {
        return Decision::ReadBodyModel;
    }
    if Some(requested) != facts.body_model.as_deref() {
        return Decision::Complete;
    }
    if !requested.starts_with("claude-router-") && !requested.to_ascii_lowercase().ends_with("[1m]")
    {
        return Decision::Complete;
    }
    if !facts.settings_loaded {
        return Decision::ReadSettings;
    }
    let result = match target(facts, requested) {
        Err(read) => return read,
        Ok(value) => value,
    };
    if facts.request_present && !facts.marked {
        return Decision::MarkNormalized;
    }
    match result {
        Some(target) => Decision::Rewrite { target },
        None => Decision::Complete,
    }
}

#[cfg(test)]
mod tests {
    use super::{Decision, Facts, decide, decoded};
    use rstest::rstest;

    #[rstest]
    #[case::canonical("claude-router-6d6f64656c", Some("model"))]
    #[case::marked("claude-router-6d6f64656c[1M]", Some("model"))]
    #[case::uppercase("claude-router-6D6F64656C", None)]
    #[case::whitespace("claude-router-6d 6f64656c", None)]
    #[case::invalid_utf8("claude-router-ff", None)]
    #[case::empty("claude-router-", None)]
    #[case::odd("claude-router-1", None)]
    #[case::unicode("claude-router-e6a8a1e59e8b", Some("模型"))]
    fn canonical_picker_decoding(#[case] name: &str, #[case] expected: Option<&str>) {
        assert_eq!(decoded(name).as_deref(), expected);
    }

    #[rstest]
    #[case::irrelevant("/models", Decision::Complete)]
    #[case::eligible("/v1/messages", Decision::ReadRouter)]
    fn route_before_reads(#[case] route: &str, #[case] expected: Decision) {
        assert_eq!(
            decide(&Facts {
                route: route.into(),
                ..Facts::default()
            }),
            expected
        );
    }
}
