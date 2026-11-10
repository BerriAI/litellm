use std::fmt;
use std::sync::Arc;

use fancy_regex::Regex;

const DIVERGENT_SYNTAX: [&str; 4] = ["&&", "--", "~~", r"\N"];

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
        if DIVERGENT_SYNTAX.iter().any(|syntax| text.contains(syntax)) {
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
                text == value || regex.is_match(value).unwrap_or_else(|_| (host.0)(value))
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

    #[test]
    fn exact_text_matches_without_asking_the_host() {
        let condition = ModelCondition::new([PatternSpec::Regex {
            text: "[[]".to_owned(),
            host: HostRegex::new(|_| panic!("an exact match never needs the host")),
        }]);
        assert!(condition.matches(Some("[[]")));
    }
}
