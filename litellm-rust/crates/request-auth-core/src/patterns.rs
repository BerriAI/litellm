use fancy_regex::Regex;
use std::{
    collections::BTreeMap,
    sync::{Arc, LazyLock, Mutex},
};

use crate::Error;

static EXPRESSIONS: LazyLock<Mutex<BTreeMap<String, Arc<Regex>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

fn expression(pattern: &str) -> Result<Arc<Regex>, Error> {
    if let Some(compiled) = EXPRESSIONS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(pattern)
    {
        return Ok(Arc::clone(compiled));
    }
    let compiled = Arc::new(Regex::new(pattern).map_err(Error::Pattern)?);
    let mut expressions = EXPRESSIONS
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if expressions.len() >= 1024 {
        expressions.clear();
    }
    expressions.insert(pattern.to_owned(), Arc::clone(&compiled));
    Ok(compiled)
}

pub fn regex_match(pattern: &str, value: &str) -> Result<bool, Error> {
    let expression = expression(pattern)?;
    Ok(expression
        .find(value)
        .map_err(Error::Pattern)?
        .is_some_and(|found| found.start() == 0))
}

pub fn model_pattern(model: &str, pattern: &str) -> Result<bool, Error> {
    if !pattern.contains('*') {
        return Ok(false);
    }
    regex_match(
        &format!("^{}(?=\\n?\\z)", pattern.replace('*', ".*")),
        model,
    )
}

pub fn route_pattern(route: &str, pattern: &str) -> Result<bool, Error> {
    let mut expression = String::from("^");
    let mut rest = pattern;
    while let Some(start) = rest.find('{') {
        expression.push_str(&rest[..start]);
        let Some(end) = rest[start + 1..].find('}').map(|index| index + start + 1) else {
            expression.push_str(&rest[start..]);
            rest = "";
            break;
        };
        let name = &rest[start + 1..end];
        if name.is_empty() {
            expression.push_str("{}");
        } else if name.ends_with(":path") {
            expression.push_str(if rest[end + 1..].contains(':') {
                "[^:]+"
            } else {
                "[\\s\\S]+"
            });
        } else {
            expression.push_str("[^/]+");
        }
        rest = &rest[end + 1..];
    }
    expression.push_str(rest);
    expression.push_str("(?=\\n?\\z)");
    regex_match(&expression, route)
}

pub fn wildcard(route: &str, pattern: &str) -> bool {
    pattern
        .strip_suffix('*')
        .map_or(route == pattern, |prefix| route.starts_with(prefix))
}

pub fn prefix(route: &str, pattern: &str) -> bool {
    route == pattern
        || route
            .strip_prefix(pattern)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

pub fn route_access(route: &str, patterns: &[String]) -> Result<bool, Error> {
    if patterns
        .iter()
        .any(|pattern| pattern == route || (pattern.ends_with('*') && wildcard(route, pattern)))
    {
        return Ok(true);
    }
    for pattern in patterns {
        if route_pattern(route, pattern)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    #[rstest]
    #[case::segment("/key/k/reset", "/key/{id}/reset", true)]
    #[case::segment_slash("/key/a/b/reset", "/key/{id}/reset", false)]
    #[case::path_slash("/credentials/a/b/start", "/credentials/{id:path}/start", true)]
    #[case::path_newline("/credentials/a\nb/start", "/credentials/{id:path}/start", true)]
    #[case::google_suffix(
        "/models/a:b:generateContent",
        "/models/{id:path}:generateContent",
        false
    )]
    #[case::final_newline("/models/a\n", "/models/{id}", true)]
    fn route_templates(#[case] route: &str, #[case] pattern: &str, #[case] expected: bool) {
        assert_eq!(super::route_pattern(route, pattern).unwrap(), expected);
    }

    #[rstest]
    #[case::provider("openai/a", "openai/*", true)]
    #[case::exact_not_pattern("a", "a", false)]
    #[case::regex_dot("aa-1", "a.-*", true)]
    #[case::newline("a\nb", "*", false)]
    #[case::final_newline("a\n", "a*", true)]
    fn model_patterns(#[case] model: &str, #[case] pattern: &str, #[case] expected: bool) {
        assert_eq!(super::model_pattern(model, pattern).unwrap(), expected);
    }
}
