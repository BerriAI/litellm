use std::ops::RangeInclusive;

#[derive(Clone, Debug)]
pub(crate) enum Single {
    Any,
    Char(char),
    Class {
        negated: bool,
        ranges: Box<[RangeInclusive<char>]>,
    },
}

impl Single {
    pub(crate) fn matches(&self, c: char) -> bool {
        match self {
            Self::Any => true,
            Self::Char(expected) => c == *expected,
            Self::Class { negated, ranges } => {
                ranges.iter().any(|range| range.contains(&c)) != *negated
            }
        }
    }
}

pub(crate) enum Token {
    Star,
    Single(Single),
}

impl Token {
    pub(crate) fn is_star(&self) -> bool {
        matches!(self, Self::Star)
    }

    pub(crate) fn single(&self) -> Option<&Single> {
        match self {
            Self::Star => None,
            Self::Single(single) => Some(single),
        }
    }
}

pub(crate) fn tokens(pattern: &[char]) -> impl Iterator<Item = Token> {
    let mut rest = pattern;
    std::iter::from_fn(move || {
        let (token, after) = next_token(rest)?;
        rest = after;
        Some(token)
    })
}

fn next_token(pattern: &[char]) -> Option<(Token, &[char])> {
    let (first, rest) = pattern.split_first()?;
    Some(match first {
        '*' => (Token::Star, rest),
        '?' => (Token::Single(Single::Any), rest),
        '[' => match class_end(rest) {
            Some(end) => (Token::Single(class(&rest[..end])), &rest[end + 1..]),
            None => (Token::Single(Single::Char('[')), rest),
        },
        other => (Token::Single(Single::Char(*other)), rest),
    })
}

/// Where the `]` closing a class sits in the text after its `[`. A `]` that comes first, or first
/// after a negating `!`, is a member.
fn class_end(after_bracket: &[char]) -> Option<usize> {
    let skip = match after_bracket {
        ['!', ']', ..] => 2,
        ['!' | ']', ..] => 1,
        _ => 0,
    };
    after_bracket[skip..]
        .iter()
        .position(|c| *c == ']')
        .map(|at| skip + at)
}

fn class(body: &[char]) -> Single {
    let (negated, members) = match body {
        ['!', members @ ..] => (true, members),
        members => (false, members),
    };
    Single::Class {
        negated,
        ranges: ranges(members),
    }
}

/// `lo-hi` is a range and any other character stands for itself. A reversed range holds nothing.
fn ranges(members: &[char]) -> Box<[RangeInclusive<char>]> {
    let mut rest = members;
    std::iter::from_fn(|| {
        let (range, after) = match rest {
            [lo, '-', hi, after @ ..] => (*lo..=*hi, after),
            [c, after @ ..] => (*c..=*c, after),
            [] => return None,
        };
        rest = after;
        Some(range)
    })
    .collect()
}
