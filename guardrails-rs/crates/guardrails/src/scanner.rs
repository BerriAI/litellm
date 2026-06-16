//! Multi-pattern text scanner used to fast-path the content-filter guardrail.
//!
//! Python's content filter screens every request by looping over each keyword
//! (`re.search(r"\b<kw>\b")`) and each regex pattern individually, under the
//! GIL. This replaces that O(keywords + patterns) work with a single
//! Aho-Corasick pass over the literal terms plus one `RegexSet` pass over the
//! patterns. The caller assigns every term a stable `id` and maps matches back
//! to its own policy (category, action, severity); this module only answers
//! "which terms matched, and where".

use aho_corasick::AhoCorasick;
use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct LiteralTerm {
    pub id: u32,
    pub text: String,
    /// Require ASCII word boundaries around the match, matching Python's
    /// `\b<kw>\b` for single-word keywords. Multi-word phrases pass `false`.
    #[serde(default)]
    pub word_boundary: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegexTerm {
    pub id: u32,
    pub pattern: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScanMatch {
    pub id: u32,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompileError {
    pub id: u32,
    pub message: String,
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80
}

fn has_word_boundaries(text: &str, start: usize, end: usize) -> bool {
    let bytes = text.as_bytes();
    let left = start == 0 || !is_word_byte(bytes[start - 1]);
    let right = end == bytes.len() || !is_word_byte(bytes[end]);
    left && right
}

pub struct Scanner {
    ac: Option<AhoCorasick>,
    literal_ids: Vec<u32>,
    literal_boundary: Vec<bool>,
    regex_set: RegexSet,
    regex_list: Vec<Regex>,
    regex_ids: Vec<u32>,
}

impl Scanner {
    /// Build a scanner from literal and regex terms. Terms that cannot be
    /// compiled (invalid regex, empty literal) are excluded and returned as
    /// `CompileError`s so the caller can fall back to its own implementation
    /// for just those terms.
    pub fn build(literals: &[LiteralTerm], regexes: &[RegexTerm]) -> (Self, Vec<CompileError>) {
        let mut errors: Vec<CompileError> = Vec::new();

        let mut literal_texts: Vec<&str> = Vec::new();
        let mut literal_ids: Vec<u32> = Vec::new();
        let mut literal_boundary: Vec<bool> = Vec::new();
        for term in literals {
            if term.text.is_empty() {
                errors.push(CompileError {
                    id: term.id,
                    message: "empty literal term".to_owned(),
                });
                continue;
            }
            literal_texts.push(&term.text);
            literal_ids.push(term.id);
            literal_boundary.push(term.word_boundary);
        }

        let ac = if literal_texts.is_empty() {
            None
        } else {
            match AhoCorasick::builder()
                .ascii_case_insensitive(true)
                .build(&literal_texts)
            {
                Ok(ac) => Some(ac),
                Err(e) => {
                    errors.push(CompileError {
                        id: 0,
                        message: format!("aho-corasick build failed: {e}"),
                    });
                    None
                }
            }
        };

        let mut regex_list: Vec<Regex> = Vec::new();
        let mut regex_ids: Vec<u32> = Vec::new();
        let mut compiled_patterns: Vec<String> = Vec::new();
        for term in regexes {
            let pattern = format!("(?i){}", term.pattern);
            match Regex::new(&pattern) {
                Ok(re) => {
                    regex_list.push(re);
                    regex_ids.push(term.id);
                    compiled_patterns.push(pattern);
                }
                Err(e) => errors.push(CompileError {
                    id: term.id,
                    message: e.to_string(),
                }),
            }
        }
        let regex_set = RegexSet::new(&compiled_patterns).unwrap_or_else(|_| RegexSet::empty());

        (
            Self {
                ac,
                literal_ids,
                literal_boundary,
                regex_set,
                regex_list,
                regex_ids,
            },
            errors,
        )
    }

    pub fn scan(&self, text: &str) -> Vec<ScanMatch> {
        let mut matches: Vec<ScanMatch> = Vec::new();

        if let Some(ac) = &self.ac {
            for m in ac.find_overlapping_iter(text) {
                let idx = m.pattern().as_usize();
                if self.literal_boundary[idx] && !has_word_boundaries(text, m.start(), m.end()) {
                    continue;
                }
                matches.push(ScanMatch {
                    id: self.literal_ids[idx],
                    start: m.start(),
                    end: m.end(),
                });
            }
        }

        if !self.regex_set.is_empty() {
            for i in self.regex_set.matches(text).into_iter() {
                for m in self.regex_list[i].find_iter(text) {
                    matches.push(ScanMatch {
                        id: self.regex_ids[i],
                        start: m.start(),
                        end: m.end(),
                    });
                }
            }
        }

        matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(id: u32, text: &str, word_boundary: bool) -> LiteralTerm {
        LiteralTerm {
            id,
            text: text.to_owned(),
            word_boundary,
        }
    }

    fn rgx(id: u32, pattern: &str) -> RegexTerm {
        RegexTerm {
            id,
            pattern: pattern.to_owned(),
        }
    }

    #[test]
    fn literal_match_is_case_insensitive() {
        let (scanner, errors) = Scanner::build(&[lit(7, "bomb", false)], &[]);
        assert!(errors.is_empty());
        let matches = scanner.scan("a BomB here");
        assert_eq!(
            matches,
            vec![ScanMatch {
                id: 7,
                start: 2,
                end: 6
            }]
        );
    }

    #[test]
    fn word_boundary_rejects_substring_match() {
        let (scanner, _) = Scanner::build(&[lit(1, "men", true)], &[]);
        // "recommend" contains "men" but not as a whole word.
        assert!(scanner.scan("we recommend this").is_empty());
        assert_eq!(scanner.scan("the men left").len(), 1);
    }

    #[test]
    fn multi_word_phrase_matches_as_substring() {
        let (scanner, _) = Scanner::build(&[lit(2, "kill all", false)], &[]);
        assert_eq!(scanner.scan("they kill all of them")[0].id, 2);
    }

    #[test]
    fn regex_returns_all_spans() {
        let (scanner, errors) = Scanner::build(&[], &[rgx(9, r"\d{3}-\d{2}-\d{4}")]);
        assert!(errors.is_empty());
        let matches = scanner.scan("a 123-45-6789 b 987-65-4321");
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| m.id == 9));
    }

    #[test]
    fn invalid_regex_is_reported_not_panicked() {
        let (scanner, errors) = Scanner::build(&[], &[rgx(3, r"(unclosed"), rgx(4, r"\d+")]);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].id, 3);
        // The valid regex still works.
        assert_eq!(scanner.scan("x 42 y")[0].id, 4);
    }

    #[test]
    fn empty_literal_is_reported() {
        let (_scanner, errors) = Scanner::build(&[lit(5, "", true)], &[]);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].id, 5);
    }

    #[test]
    fn literals_and_regexes_combine() {
        let (scanner, _) = Scanner::build(&[lit(1, "ssn", false)], &[rgx(2, r"\d{3}-\d{2}-\d{4}")]);
        let ids: Vec<u32> = scanner
            .scan("ssn 123-45-6789")
            .iter()
            .map(|m| m.id)
            .collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }
}
