mod error;

pub use error::InvalidBoolean;

fn python_trim(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}

pub fn parse_env_bool(value: &str) -> Result<bool, InvalidBoolean> {
    let token = python_trim(value);
    if ["1", "true", "t", "yes", "y", "on"]
        .iter()
        .any(|candidate| token.eq_ignore_ascii_case(candidate))
    {
        return Ok(true);
    }
    if ["0", "false", "f", "no", "n", "off"]
        .iter()
        .any(|candidate| token.eq_ignore_ascii_case(candidate))
    {
        return Ok(false);
    }
    Err(InvalidBoolean)
}

pub fn parse_str_bool(value: &str) -> Option<bool> {
    let token = python_trim(value);
    if token.eq_ignore_ascii_case("true") {
        return Some(true);
    }
    token.eq_ignore_ascii_case("false").then_some(false)
}

pub fn parse_redis_bool(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
}
