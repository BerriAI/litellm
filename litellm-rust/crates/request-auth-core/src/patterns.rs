use fancy_regex::Regex;
use std::{
    collections::BTreeMap,
    sync::{Arc, LazyLock, Mutex},
};

use crate::Error;

static EXPRESSIONS: LazyLock<Mutex<BTreeMap<String, Arc<Regex>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

fn python_escape(escaped: char, in_class: bool) -> Option<String> {
    let word = r"\p{L}\p{N}_";
    let class = match escaped {
        'w' => format!("[{word}]"),
        'W' => format!("[^{word}]"),
        'b' if !in_class => format!("(?:(?<=[{word}])(?![{word}])|(?<![{word}])(?=[{word}]))"),
        'B' if !in_class => {
            format!("(?:(?<=[{word}])(?=[{word}])|(?<![{word}])(?![{word}]))(?!\\A\\z)")
        }
        _ => return None,
    };
    Some(format!("(?-i:{class})"))
}

fn python_class(pattern: &str) -> Option<(usize, String)> {
    let negated = pattern.starts_with("[^");
    let start = if negated { 2 } else { 1 };
    let mut chars = pattern.char_indices().skip(start).peekable();
    let mut literals = String::new();
    let mut alternatives = Vec::new();
    let mut range = false;
    if chars.peek().is_some_and(|(_, ch)| *ch == ']') {
        literals.push_str(r"\]");
        chars.next();
    }
    while let Some((index, ch)) = chars.next() {
        match ch {
            ']' => {
                if alternatives.is_empty() {
                    return Some((index + 1, pattern[..index + 1].into()));
                }
                if !literals.is_empty() {
                    alternatives.push(format!("[{literals}]"));
                }
                let union = alternatives.join("|");
                return Some((
                    index + 1,
                    if negated {
                        format!("(?:(?!(?:{union}))[\\s\\S])")
                    } else {
                        format!("(?:{union})")
                    },
                ));
            }
            '\\' => {
                let (_, escaped) = chars.next()?;
                if let Some(class) = python_escape(escaped, true) {
                    // A category escape cannot be a range endpoint in Python.
                    if range
                        || pattern[index + 2..].starts_with('-')
                            && !pattern[index + 2..].starts_with("-]")
                    {
                        return None;
                    }
                    alternatives.push(class);
                } else {
                    literals.push(ch);
                    literals.push(escaped);
                }
            }
            '[' => literals.push_str(r"\["),
            '^' => literals.push_str(r"\^"),
            _ => literals.push(ch),
        }
        range = ch == '-' && index > start;
    }
    None
}

fn python_pattern(pattern: &str) -> String {
    let mut output = String::with_capacity(pattern.len());
    let mut verbose = false;
    let mut scopes = Vec::new();
    let mut chars = pattern.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        let rest = &pattern[index..];
        match ch {
            '\\' => {
                if let Some((_, escaped)) = chars.next() {
                    if let Some(replacement) = python_escape(escaped, false) {
                        output.push_str(&replacement);
                    } else {
                        output.push(ch);
                        output.push(escaped);
                    }
                } else {
                    output.push(ch);
                }
            }
            '[' => {
                let Some((length, class)) = python_class(rest) else {
                    output.push_str(rest);
                    break;
                };
                output.push_str(&class);
                while chars.peek().is_some_and(|(next, _)| *next < index + length) {
                    chars.next();
                }
            }
            '#' if verbose => {
                output.push(ch);
                for (_, next) in chars.by_ref() {
                    output.push(next);
                    if next == '\n' {
                        break;
                    }
                }
            }
            '(' if rest.starts_with("(?#") => {
                output.push(ch);
                let mut escaped = false;
                for (_, next) in chars.by_ref() {
                    output.push(next);
                    if next == ')' && !escaped {
                        break;
                    }
                    escaped = next == '\\' && !escaped;
                }
            }
            '(' => {
                if let Some(header) = rest.strip_prefix("(?") {
                    let length = header
                        .find(|c: char| !"aiLmsux-".contains(c))
                        .unwrap_or(header.len());
                    if length > 0 && matches!(header.as_bytes().get(length), Some(b':' | b')')) {
                        let scoped = header.as_bytes()[length] == b':';
                        if scoped {
                            scopes.push(verbose);
                        }
                        let mut disabled = false;
                        for flag in header[..length].chars() {
                            match flag {
                                'x' => verbose = !disabled,
                                '-' => disabled = true,
                                _ => (),
                            }
                        }
                        output.push_str(&rest[..length + 3]);
                        while chars
                            .peek()
                            .is_some_and(|(next, _)| *next < index + length + 3)
                        {
                            chars.next();
                        }
                        continue;
                    }
                }
                scopes.push(verbose);
                output.push(ch);
            }
            ')' => {
                verbose = scopes.pop().unwrap_or_default();
                output.push(ch);
            }
            _ => output.push(ch),
        }
    }
    output
}

fn expression(pattern: &str) -> Result<Arc<Regex>, Error> {
    if let Some(compiled) = EXPRESSIONS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(pattern)
    {
        return Ok(Arc::clone(compiled));
    }
    let compiled = Arc::new(Regex::new(&python_pattern(pattern)).map_err(Error::Pattern)?);
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
    #[case::combining(r"\w+", "\u{0301}a", false)]
    #[case::joiner(r"\w+", "\u{200d}a", false)]
    #[case::connector(r"\w+", "\u{203f}a", false)]
    #[case::superscript(r"\w+", "²a", true)]
    #[case::fraction(r"\w+", "¼", true)]
    #[case::letter_number(r"\w+", "Ⅷ", true)]
    #[case::underscore(r"\w+", "_", true)]
    #[case::letter(r"\w+", "한", true)]
    #[case::nonword_mark(r"\W+", "\u{0301}a", true)]
    #[case::nonword_number(r"\W+", "²a", false)]
    #[case::class_word(r"[\w]+", "²a", true)]
    #[case::class_nonword(r"[\W]+", "\u{0301}a", true)]
    #[case::negated_word(r"[^\w]+", "²a", false)]
    #[case::negated_nonword(r"[^\W]+", "²a", true)]
    #[case::negated_repeat(r"[^\w]+$", "!a", false)]
    #[case::union(r"[\w\W]+", "\u{0301}a", true)]
    #[case::negated_union(r"[^\w\W]+", "a", false)]
    #[case::class_caret(r"[\w^]+", "^", true)]
    #[case::class_dash(r"[\w-]+", "-", true)]
    #[case::class_escaped_dash(r"[\-\w]+", "²", true)]
    #[case::class_closing_bracket(r"[]\w]+", "]²", true)]
    #[case::class_opening_bracket(r"[[\w]+", "[²", true)]
    #[case::class_backspace(r"[\b\w]", "\u{8}", true)]
    #[case::literal_escape(r"\\w", r"\w", true)]
    #[case::literal_class_escape(r"[\\w]", "²", false)]
    #[case::boundary_number(r"\b²", "²", true)]
    #[case::boundary_mark(r"\b\u0301", "\u{0301}", false)]
    #[case::nonboundary_mark(r"\B\u0301", "\u{0301}", true)]
    #[case::nonboundary_number(r"\B²", "²", false)]
    #[case::boundary_after_word(r"a\b\u0301", "a\u{0301}", true)]
    #[case::boundary_after_nonword(r"!\b²", "!²", true)]
    #[case::nonboundary_between_words(r"a\B²", "a²", true)]
    #[case::nonboundary_between_nonwords(r"!\B\u0301", "!\u{0301}", true)]
    #[case::end_boundary(r"²\b", "²", true)]
    #[case::end_nonboundary(r"\u0301\B", "\u{0301}", true)]
    #[case::empty_boundary(r"\b", "", false)]
    #[case::empty_nonboundary(r"\B", "", false)]
    #[case::unicode_flag(r"(?u:\w)", "²", true)]
    #[case::case_insensitive_word(r"(?i:\w)", "\u{0345}", false)]
    #[case::case_insensitive_class_word(r"(?i:[\w])", "\u{0345}", false)]
    #[case::case_insensitive_class_literal(r"(?i:[\wΙ])", "\u{0345}", true)]
    #[case::case_insensitive_boundary(r"(?i:\b\u0345)", "\u{0345}", false)]
    #[case::case_insensitive_literal(r"(?i:a)\w", "A²", true)]
    #[case::ascii_class(r"[a-zA-Z0-9_]", "²", false)]
    #[case::comment(r"(?# [\b)\w", "²", true)]
    #[case::verbose_comment("(?x:# [\\b)\n\\w)", "²", true)]
    #[case::verbose_scope("(?x:# [\n)(?-x:#)\\w", "#²", true)]
    #[case::verbose_restored("(?x:# [\n)#\\w", "#²", true)]
    #[case::backreference(r"(\w)\1", "²²", true)]
    #[case::lookbehind(r".(?<=[\w])a", "²a", true)]
    fn python_word_semantics(#[case] pattern: &str, #[case] value: &str, #[case] expected: bool) {
        assert_eq!(super::regex_match(pattern, value).unwrap(), expected);
    }

    #[rstest]
    #[case::ascii_flag(r"(?a:\w)")]
    #[case::ascii_unicode_flag(r"(?-u:\w)")]
    #[case::word_range_start(r"[\w-a]")]
    #[case::word_range_end(r"[a-\w]")]
    #[case::unclosed_class(r"[\w")]
    #[case::unclosed_group(r"(\w")]
    fn invalid_patterns_remain_errors(#[case] pattern: &str) {
        assert!(super::regex_match(pattern, "a").is_err());
    }

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
    #[case::word_combining_mark("\u{0301}a", r"\w*", false)]
    #[case::word_join_control("\u{200d}a", r"\w*", false)]
    #[case::word_other_number("\u{00b2}a", r"\w*", true)]
    fn model_patterns(#[case] model: &str, #[case] pattern: &str, #[case] expected: bool) {
        assert_eq!(super::model_pattern(model, pattern).unwrap(), expected);
    }
}
