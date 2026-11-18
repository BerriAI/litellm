//! Where this port is stricter than PyYAML on purpose: a `Value` copies every alias, and
//! construction recurses, so expanded depth and size are capped.

use litellm_python_compat::MAX_DEPTH;
use litellm_yaml::{Error, from_str};
use rstest::rstest;

#[derive(Debug, PartialEq)]
enum Outcome {
    Loads,
    TooDeep,
    TooLarge,
}

fn outcome(text: &str) -> Outcome {
    match from_str(text) {
        Ok(_) => Outcome::Loads,
        Err(Error::TooDeep { .. }) => Outcome::TooDeep,
        Err(Error::TooLarge { .. }) => Outcome::TooLarge,
        Err(error) => panic!("{text:?} failed for another reason: {error}"),
    }
}

/// A scalar `depth` levels down, in flow sequences.
fn flow(depth: usize) -> String {
    format!("{}x{}", "[".repeat(depth), "]".repeat(depth))
}

/// Each anchor wraps the previous one in a collection, so every line nests two levels deep while
/// the last one expands to a scalar `depth` levels down.
fn alias_chain(depth: usize, wrap: fn(String) -> String) -> String {
    let links: String = (2..=depth)
        .map(|link| format!("- &a{link} {}\n", wrap(format!("*a{}", link - 1))))
        .collect();
    format!("- &a1 x\n{links}")
}

fn in_sequence(alias: String) -> String {
    format!("[{alias}]")
}

fn in_mapping(alias: String) -> String {
    format!("{{k: {alias}}}")
}

/// `a` is a sequence of sequences anchored at the top, aliased `depth - 2` levels down, so the
/// alias itself sits within the limit while its scalars end up `depth` levels down.
fn alias_under_nesting(depth: usize) -> String {
    let at = depth - 2;
    format!(
        "a: &a [[x]]\nb: {}*a{}",
        "[".repeat(at - 1),
        "]".repeat(at - 1)
    )
}

/// Each level aliases the previous one ten times, so seven levels expand to ten million nodes.
fn aliases_multiplied() -> String {
    let levels: String = (1..7)
        .map(|level| {
            let previous = format!("*l{}", level - 1);
            format!("l{level}: &l{level} [{}]\n", vec![previous; 10].join(", "))
        })
        .collect();
    format!("l0: &l0 [x, x, x, x, x, x, x, x, x, x]\n{levels}")
}

#[rstest]
#[case::flow_at_the_limit(flow(MAX_DEPTH), Outcome::Loads)]
#[case::flow_past_the_limit(flow(MAX_DEPTH + 1), Outcome::TooDeep)]
#[case::empty_flow_at_the_limit(format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1)), Outcome::Loads)]
#[case::empty_flow_past_the_limit(format!("{}{}", "[".repeat(MAX_DEPTH + 2), "]".repeat(MAX_DEPTH + 2)), Outcome::TooDeep)]
#[case::sequence_alias_chain_at_the_limit(alias_chain(MAX_DEPTH, in_sequence), Outcome::Loads)]
#[case::sequence_alias_chain_past_the_limit(alias_chain(MAX_DEPTH + 1, in_sequence), Outcome::TooDeep)]
#[case::mapping_alias_chain_at_the_limit(alias_chain(MAX_DEPTH, in_mapping), Outcome::Loads)]
#[case::mapping_alias_chain_past_the_limit(alias_chain(MAX_DEPTH + 1, in_mapping), Outcome::TooDeep)]
#[case::alias_under_nesting_at_the_limit(alias_under_nesting(MAX_DEPTH), Outcome::Loads)]
#[case::alias_under_nesting_past_the_limit(alias_under_nesting(MAX_DEPTH + 1), Outcome::TooDeep)]
#[case::aliases_multiplied(aliases_multiplied(), Outcome::TooLarge)]
fn expansion_is_capped(#[case] text: String, #[case] expected: Outcome) {
    assert_eq!(outcome(&text), expected);
}
