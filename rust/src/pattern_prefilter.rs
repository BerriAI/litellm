use std::collections::HashSet;

use pyo3::pyclass;
use pyo3::pymethods;
use regex::{RegexBuilder, RegexSet, RegexSetBuilder};

/// Conservative "could any of these patterns possibly match?" gate.
///
/// Built from a fixed set of regex sources that each compiled successfully on
/// their own. `any_match` defaults to `true` (don't skip) whenever there is no
/// usable set, since the only safe direction to fail in is "check anyway."
#[pyclass]
pub struct PatternPrefilter {
    regex_set: Option<RegexSet>,
}

#[pymethods]
impl PatternPrefilter {
    fn any_match(&self, text: &str) -> bool {
        self.regex_set.as_ref().is_none_or(|set| set.is_match(text))
    }
}

/// Splits `patterns` into ones a `RegexSet` can evaluate and ones it can't
/// (e.g. lookaround/backreferences, which Rust's `regex` crate intentionally
/// doesn't support), then builds a single combined automaton from the former.
///
/// Returns the prefilter plus the indices of `patterns` it does not cover;
/// callers must keep running those through the existing per-pattern path.
#[pyo3::pyfunction]
pub fn build_pattern_prefilter(patterns: Vec<String>) -> (PatternPrefilter, Vec<usize>) {
    let mut covered_sources: Vec<&str> = Vec::new();
    let mut rejected_indices: Vec<usize> = Vec::new();

    for (index, source) in patterns.iter().enumerate() {
        match RegexBuilder::new(source).case_insensitive(true).build() {
            Ok(_) => covered_sources.push(source),
            Err(_) => rejected_indices.push(index),
        }
    }

    if covered_sources.is_empty() {
        return (PatternPrefilter { regex_set: None }, rejected_indices);
    }

    match RegexSetBuilder::new(&covered_sources)
        .case_insensitive(true)
        .build()
    {
        Ok(set) => (
            PatternPrefilter {
                regex_set: Some(set),
            },
            rejected_indices,
        ),
        Err(_) => {
            let already_rejected: HashSet<usize> = rejected_indices.iter().copied().collect();
            rejected_indices.extend((0..patterns.len()).filter(|i| !already_rejected.contains(i)));
            (PatternPrefilter { regex_set: None }, rejected_indices)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::pattern_prefilter::build_pattern_prefilter;

    #[test]
    fn empty_pattern_list_defaults_to_match_anything() {
        let (prefilter, rejected) = build_pattern_prefilter(vec![]);
        assert!(rejected.is_empty());
        assert!(prefilter.any_match("anything at all"));
    }

    #[test]
    fn matching_is_case_insensitive() {
        let (prefilter, _) = build_pattern_prefilter(vec!["secret".into()]);
        assert!(prefilter.any_match("this is a SECRET"));
    }

    #[test]
    fn lookbehind_pattern_is_rejected_by_index_and_others_still_build() {
        let (prefilter, rejected) =
            build_pattern_prefilter(vec!["\\d{3}-\\d{2}-\\d{4}".into(), "(?<=\\$)\\d+".into()]);
        assert_eq!(rejected, vec![1]);
        assert!(prefilter.any_match("123-45-6789"));
    }

    #[test]
    fn rejected_pattern_does_not_suppress_matches_on_covered_ones() {
        let (prefilter, rejected) =
            build_pattern_prefilter(vec!["(?<=\\$)\\d+".into(), "\\d{3}-\\d{2}-\\d{4}".into()]);
        assert_eq!(rejected, vec![0]);
        assert!(prefilter.any_match("123-45-6789"));
        assert!(!prefilter.any_match("no match here"));
    }

    #[test]
    fn all_patterns_rejected_defaults_to_match_anything() {
        let (prefilter, rejected) = build_pattern_prefilter(vec!["(?<=\\$)\\d+".into()]);
        assert_eq!(rejected, vec![0]);
        assert!(prefilter.any_match("anything at all"));
    }

    #[test]
    fn interleaved_covered_and_rejected_indices_preserve_order() {
        let (prefilter, rejected) = build_pattern_prefilter(vec![
            "(?<=\\$)\\d+".into(),
            "alpha".into(),
            "(?<=#)\\d+".into(),
            "beta".into(),
            "(?<=@)\\d+".into(),
        ]);
        assert_eq!(rejected, vec![0, 2, 4]);
        assert!(prefilter.any_match("alpha"));
        assert!(prefilter.any_match("beta"));
        assert!(!prefilter.any_match("gamma"));
    }

    #[test]
    fn regex_set_exceeding_size_limit_defaults_to_match_anything() {
        // Each of these compiles fine on its own (RegexBuilder succeeds), but
        // combining 14 of them into a single RegexSet overflows the crate's
        // default 10MB compiled-program size limit, exercising the fallback
        // that's distinct from "no pattern was individually valid."
        let oversized = "a{1,4000}b{1,4000}c{1,4000}";
        let mut sources: Vec<String> = vec![oversized.into(); 14];
        sources.push("simple".into());
        let expected_rejected_count = sources.len();
        let (prefilter, rejected) = build_pattern_prefilter(sources);
        assert_eq!(rejected.len(), expected_rejected_count);
        assert!(prefilter.any_match("anything at all"));
    }
}
