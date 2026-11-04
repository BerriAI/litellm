use sha1::{Digest, Sha1};

pub(crate) const SAMPLE_SPEC: &str = "sample_spec";
pub(crate) const FALLBACK_GENERALIZATIONS: &str = "fallback_generalizations";

/// Why a fetched map is too small to serve
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SizeProblem {
    BelowMinimum {
        actual: usize,
        minimum: usize,
    },
    Shrunk {
        actual: usize,
        reference: usize,
        ratio: f64,
    },
}

/// Whether a top-level key is a model, as opposed to a document section
pub fn is_model_key(key: &str) -> bool {
    key != SAMPLE_SPEC && key != FALLBACK_GENERALIZATIONS
}

/// `actual` models against `minimum` and against `ratio` of the `reference` map's models, skipped for an empty reference
pub fn size_problem(
    actual: usize,
    reference: usize,
    minimum: usize,
    ratio: f64,
) -> Option<SizeProblem> {
    if actual < minimum {
        return Some(SizeProblem::BelowMinimum { actual, minimum });
    }
    (reference > 0 && (actual as f64) < (reference as f64) * ratio).then_some(SizeProblem::Shrunk {
        actual,
        reference,
        ratio,
    })
}

/// The git blob id of a document's bytes, which is the revision a host reports for it
pub fn revision(body: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", body.len()).as_bytes());
    hasher.update(body);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::model("gpt-4o", true)]
    #[case::sample_spec("sample_spec", false)]
    #[case::fallback_generalizations("fallback_generalizations", false)]
    #[case::case_sensitive("Sample_Spec", true)]
    fn sections_are_not_models(#[case] key: &str, #[case] expected: bool) {
        assert_eq!(is_model_key(key), expected);
    }

    #[rstest]
    #[case::fits(10, 10, 5, 0.5, None)]
    #[case::below_minimum(4, 10, 5, 0.5, Some(SizeProblem::BelowMinimum { actual: 4, minimum: 5 }))]
    #[case::minimum_checked_first(1, 10, 5, 0.5, Some(SizeProblem::BelowMinimum { actual: 1, minimum: 5 }))]
    #[case::shrunk(5, 11, 0, 0.5, Some(SizeProblem::Shrunk { actual: 5, reference: 11, ratio: 0.5 }))]
    #[case::exactly_at_ratio(5, 10, 0, 0.5, None)]
    #[case::empty_reference_skips_ratio(0, 0, 0, 2.0, None)]
    #[case::ratio_above_one(10, 10, 0, 1.5, Some(SizeProblem::Shrunk { actual: 10, reference: 10, ratio: 1.5 }))]
    #[case::nan_ratio_never_shrinks(0, 10, 0, f64::NAN, None)]
    fn size_matches_python_validate_model_cost_map(
        #[case] actual: usize,
        #[case] reference: usize,
        #[case] minimum: usize,
        #[case] ratio: f64,
        #[case] expected: Option<SizeProblem>,
    ) {
        assert_eq!(size_problem(actual, reference, minimum, ratio), expected);
    }

    #[rstest]
    fn the_revision_is_the_git_blob_id_of_the_body() {
        assert_eq!(
            revision(b"hello\n"),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
    }
}
