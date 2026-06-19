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
    let mut covered_indices: Vec<usize> = Vec::new();
    let mut rejected_indices: Vec<usize> = Vec::new();

    for (index, source) in patterns.iter().enumerate() {
        match RegexBuilder::new(source).case_insensitive(true).build() {
            Ok(_) => {
                covered_sources.push(source);
                covered_indices.push(index);
            }
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
            rejected_indices.extend(covered_indices);
            (PatternPrefilter { regex_set: None }, rejected_indices)
        }
    }
}
