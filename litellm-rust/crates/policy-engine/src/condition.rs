use std::fmt;
use std::sync::Arc;

use fancy_regex::Regex;

const ESCAPES: &[u8] = br".^$*+?{}[]|()\-/dDwWsSbBA";
const CLASS_ESCAPES: &[u8] = br".^$*+?{}[]|()\-/dDwWsS";
const SCOPED_FLAGS: &[u8] = b"ims";

/// Whether fancy-regex reads `pattern` exactly as Python's `re` does on [`plain_ascii`] values.
/// Only a vetted subset qualifies: printable ASCII, the metacharacters, the escapes above, counted
/// repeats with digits only, non-possessive quantifiers, `(?:`, the `i`/`m`/`s` flags and classes
/// without nesting, set operations or a leading `]`. Everything else is matched by Python.
fn read_alike(pattern: &str) -> bool {
    let bytes = pattern.as_bytes();
    if !bytes.iter().all(u8::is_ascii_graphic_or_space) {
        return false;
    }
    let mut at = 0;
    let mut after_quantifier = false;
    while let Some(&byte) = bytes.get(at) {
        let (width, quantifier) = match byte {
            b'\\' => match bytes.get(at + 1) {
                Some(escaped) if ESCAPES.contains(escaped) => (2, false),
                _ => return false,
            },
            b'[' => match class_width(&bytes[at..]) {
                Some(width) => (width, false),
                None => return false,
            },
            b'(' if bytes.get(at + 1) == Some(&b'?') => match flags_width(&bytes[at + 2..]) {
                Some(width) => (width + 2, false),
                None => return false,
            },
            b'{' => match repeat_width(&bytes[at..]) {
                Some(width) => (width, true),
                None => return false,
            },
            b'+' if after_quantifier => return false,
            b'*' | b'+' | b'?' => (1, true),
            _ => (1, false),
        };
        at += width;
        after_quantifier = quantifier;
    }
    true
}

trait GraphicOrSpace {
    fn is_ascii_graphic_or_space(&self) -> bool;
}

impl GraphicOrSpace for u8 {
    fn is_ascii_graphic_or_space(&self) -> bool {
        self.is_ascii_graphic() || *self == b' '
    }
}

fn class_width(class: &[u8]) -> Option<usize> {
    let mut at = if class.get(1) == Some(&b'^') { 2 } else { 1 };
    if class.get(at) == Some(&b']') {
        return None;
    }
    loop {
        match *class.get(at)? {
            b']' => return Some(at + 1),
            b'[' => return None,
            b'\\' if CLASS_ESCAPES.contains(class.get(at + 1)?) => at += 2,
            b'\\' => return None,
            operator @ (b'&' | b'-' | b'~') if class.get(at + 1) == Some(&operator) => return None,
            _ => at += 1,
        }
    }
}

fn flags_width(group: &[u8]) -> Option<usize> {
    let flags = group
        .iter()
        .take_while(|byte| SCOPED_FLAGS.contains(byte))
        .count();
    match group.get(flags)? {
        b':' | b')' => Some(flags + 1),
        _ => None,
    }
}

fn repeat_width(repeat: &[u8]) -> Option<usize> {
    let body = &repeat[1..];
    let close = body.iter().position(|&byte| byte == b'}')?;
    let (low, high) = match body[..close].iter().position(|&byte| byte == b',') {
        Some(comma) => (&body[..comma], &body[comma + 1..close]),
        None => (&body[..close], &b""[..]),
    };
    let digits = |part: &[u8]| part.iter().all(u8::is_ascii_digit);
    (!low.is_empty() && digits(low) && digits(high)).then_some(close + 2)
}

/// Values the read-alike subset matches the same way in both engines. Python's `$` also matches
/// before a final newline, its `\s` covers `\x1c`-`\x1f`, its `\B` never matches an empty string
/// and its Unicode tables differ from Rust's, so anything else is matched by Python.
fn plain_ascii(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_graphic_or_space())
}

#[derive(Clone)]
pub struct HostRegex(Arc<dyn Fn(&str) -> bool + Send + Sync>);

impl HostRegex {
    pub fn new(full_match: impl Fn(&str) -> bool + Send + Sync + 'static) -> Self {
        Self(Arc::new(full_match))
    }
}

impl fmt::Debug for HostRegex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HostRegex")
    }
}

#[derive(Debug, Clone)]
pub enum PatternSpec {
    Literal(String),
    Regex { text: String, host: HostRegex },
}

#[derive(Debug)]
enum Pattern {
    Literal(String),
    Regex {
        text: String,
        regex: Regex,
        host: HostRegex,
    },
    Host {
        text: String,
        host: HostRegex,
    },
}

impl Pattern {
    fn new(spec: PatternSpec) -> Self {
        let (text, host) = match spec {
            PatternSpec::Literal(text) => return Self::Literal(text),
            PatternSpec::Regex { text, host } => (text, host),
        };
        if !read_alike(&text) {
            return Self::Host { text, host };
        }
        match Regex::new(&format!(r"\A(?:{text})\z")) {
            Ok(regex) => Self::Regex { text, regex, host },
            Err(_) => Self::Host { text, host },
        }
    }

    fn matches(&self, value: &str) -> bool {
        match self {
            Self::Literal(text) => text == value,
            Self::Regex { text, regex, host } => {
                text == value
                    || if plain_ascii(value) {
                        regex.is_match(value).unwrap_or_else(|_| (host.0)(value))
                    } else {
                        (host.0)(value)
                    }
            }
            Self::Host { text, host } => text == value || (host.0)(value),
        }
    }
}

#[derive(Debug)]
pub struct ModelCondition(Vec<Pattern>);

impl ModelCondition {
    pub fn new(patterns: impl IntoIterator<Item = PatternSpec>) -> Self {
        Self(patterns.into_iter().map(Pattern::new).collect())
    }

    pub fn matches(&self, model: Option<&str>) -> bool {
        model.is_some_and(|model| self.0.iter().any(|pattern| pattern.matches(model)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use rstest::rstest;

    use super::*;

    fn regex(text: &str) -> PatternSpec {
        PatternSpec::Regex {
            text: text.to_owned(),
            host: HostRegex::new(|_| panic!("Rust reads this pattern itself")),
        }
    }

    #[rstest]
    #[case::exact(vec![regex("gpt-4")], Some("gpt-4"), true)]
    #[case::regex_is_anchored_at_both_ends(vec![regex("gpt-4")], Some("gpt-4o"), false)]
    #[case::regex_full_match(vec![regex("gpt-4.*")], Some("gpt-4o-mini"), true)]
    #[case::regex_needs_whole_value(vec![regex("gpt")], Some("my-gpt"), false)]
    #[case::alternation_backtracks_to_full_match(vec![regex("a|ab")], Some("ab"), true)]
    #[case::scoped_inline_flag(vec![regex("(?i)GPT-4")], Some("gpt-4"), true)]
    #[case::literal_is_exact(vec![PatternSpec::Literal("gpt-4(".to_owned())], Some("gpt-4("), true)]
    #[case::literal_is_never_regex(vec![PatternSpec::Literal("abc(?i)".to_owned())], Some("ABC"), false)]
    #[case::any_of_list(vec![regex("claude-.*"), regex("gpt-4")], Some("gpt-4"), true)]
    #[case::empty_list_never_matches(vec![], Some("gpt-4"), false)]
    #[case::missing_model_never_matches(vec![regex(".*")], None, false)]
    fn matches_like_python_fullmatch(
        #[case] patterns: Vec<PatternSpec>,
        #[case] model: Option<&str>,
        #[case] expected: bool,
    ) {
        assert_eq!(ModelCondition::new(patterns).matches(model), expected);
    }

    #[rstest]
    #[case::nested_set("[[]")]
    #[case::ascii_flag(r"(?a)\w")]
    #[case::verbose_comment_swallows_anchor("(?x) a b # c")]
    #[case::set_intersection("[a&&b]")]
    #[case::set_difference("[a--b]")]
    #[case::symmetric_difference("[a~~b]")]
    #[case::named_character(r"\N{LATIN CAPITAL LETTER A}")]
    #[case::posix_class("[[:alpha:]]")]
    #[case::word_start_escape(r"\<a")]
    #[case::word_end_escape(r"a\>")]
    #[case::word_boundary_form(r"\b{start}a")]
    #[case::class_opening_with_bracket_range("[]-z]")]
    #[case::negated_class_opening_with_bracket("[^]a]")]
    #[case::octal_escape_after_a_group(r"(a)\01")]
    #[case::backreference(r"(a)\1")]
    #[case::verbose_counted_repeat("(?x).{ 1 }")]
    #[case::possessive_counted_repeat("(?:a+){2}+")]
    #[case::possessive_repeat("a++")]
    #[case::counted_repeat_without_minimum("a{,2}")]
    #[case::lookahead("a(?=b)")]
    #[case::named_group("(?P<n>a)")]
    #[case::absolute_end(r"a\Z")]
    #[case::non_ascii_pattern("é.*")]
    #[case::newline_in_pattern("a$\n")]
    fn patterns_rust_reads_differently_go_to_the_host(#[case] text: &str) {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let condition = ModelCondition::new([PatternSpec::Regex {
            text: text.to_owned(),
            host: HostRegex::new(move |value| {
                seen.fetch_add(1, Ordering::SeqCst);
                value == "host says yes"
            }),
        }]);
        assert!(condition.matches(Some("host says yes")));
        assert!(!condition.matches(Some("anything else")));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[rstest]
    #[case::information_separator(r"a\sb", "a\x1cb")]
    #[case::non_ascii_value(r"a\wb", "a\u{b2}b")]
    #[case::case_folding(r"(?i)i", "\u{131}")]
    #[case::trailing_newline("a$", "a\n")]
    #[case::empty_value(r"\B", "")]
    fn values_the_engines_read_differently_go_to_the_host(#[case] text: &str, #[case] value: &str) {
        let condition = ModelCondition::new([PatternSpec::Regex {
            text: text.to_owned(),
            host: HostRegex::new(|value| value != "a"),
        }]);
        assert!(condition.matches(Some(value)));
    }

    #[rstest]
    #[case::scoped_flags("(?i:gpt)-4", "GPT-4")]
    #[case::global_flags("(?ims)gpt.*", "GPT-4o")]
    #[case::counted_repeats(r"gpt-\d{1,2}", "gpt-42")]
    #[case::class_with_escapes(r"[\w.\-]+", "gpt-4.1")]
    #[case::lazy_quantifier("gpt.*?o", "gpt-4o")]
    #[case::word_boundary(r"gpt\b.*", "gpt 4")]
    fn read_alike_patterns_stay_in_rust(#[case] text: &str, #[case] value: &str) {
        assert!(ModelCondition::new([regex(text)]).matches(Some(value)));
    }

    #[test]
    fn exact_text_matches_without_asking_the_host() {
        let condition = ModelCondition::new([PatternSpec::Regex {
            text: "[[]".to_owned(),
            host: HostRegex::new(|_| panic!("an exact match never needs the host")),
        }]);
        assert!(condition.matches(Some("[[]")));
    }
}
