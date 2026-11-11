#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Star,
    Any,
    Char(char),
    Class {
        negated: bool,
        items: Vec<(char, char)>,
    },
    Never,
}

impl Token {
    fn matches(&self, c: char) -> bool {
        match self {
            Self::Star | Self::Any => true,
            Self::Char(expected) => c == *expected,
            Self::Class { negated, items } => {
                items.iter().any(|(lo, hi)| (*lo..=*hi).contains(&c)) != *negated
            }
            Self::Never => false,
        }
    }
}

/// A case-sensitive `fnmatch` glob with `*`, `?` and `[...]` classes, `!` negating a class,
/// compiled once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Pattern(Box<[Token]>);

impl Pattern {
    pub(crate) fn new(pattern: &str) -> Self {
        Self(tokens(pattern).into())
    }

    pub(crate) fn matches(&self, name: &str) -> bool {
        let name: Vec<char> = name.chars().collect();
        glob_matches(&self.0, &name)
    }
}

pub(crate) fn glob_match(name: &str, pattern: &str) -> bool {
    Pattern::new(pattern).matches(name)
}

fn glob_matches(globs: &[Token], name: &[char]) -> bool {
    match globs.split_first() {
        None => name.is_empty(),
        Some((Token::Star, rest)) => {
            (0..=name.len()).any(|start| glob_matches(rest, &name[start..]))
        }
        Some((glob, rest)) => name
            .split_first()
            .is_some_and(|(c, tail)| glob.matches(*c) && glob_matches(rest, tail)),
    }
}

/// `fnmatch.translate`, kept as tokens instead of a regex.
fn tokens(pattern: &str) -> Vec<Token> {
    let pat: Vec<char> = pattern.chars().collect();
    let n = pat.len();
    let mut globs = Vec::new();
    let mut i = 0;
    while i < n {
        let c = pat[i];
        i += 1;
        match c {
            '*' if globs.last() == Some(&Token::Star) => {}
            '*' => globs.push(Token::Star),
            '?' => globs.push(Token::Any),
            '[' => {
                let mut j = i;
                if j < n && pat[j] == '!' {
                    j += 1;
                }
                if j < n && pat[j] == ']' {
                    j += 1;
                }
                while j < n && pat[j] != ']' {
                    j += 1;
                }
                if j >= n {
                    globs.push(Token::Char('['));
                } else {
                    globs.push(class(&class_source(&pat, i, j)));
                    i = j + 1;
                }
            }
            other => globs.push(Token::Char(other)),
        }
    }
    globs
}

fn escape_class_chars(chunk: &[char], hyphens: bool) -> Vec<char> {
    chunk
        .iter()
        .flat_map(|c| match c {
            '\\' | '&' | '~' | '|' => vec!['\\', *c],
            '-' if hyphens => vec!['\\', '-'],
            other => vec![*other],
        })
        .collect()
}

/// The regex class body `fnmatch.translate` writes for `pat[start..end]`.
fn class_source(pat: &[char], start: usize, end: usize) -> Vec<char> {
    let body = &pat[start..end];
    if !body.contains(&'-') {
        return escape_class_chars(body, false);
    }
    let mut chunks: Vec<Vec<char>> = Vec::new();
    let mut i = start;
    let mut k = if pat[start] == '!' {
        start + 2
    } else {
        start + 1
    };
    while let Some(found) = (k..end).find(|index| pat[*index] == '-') {
        chunks.push(pat[i..found].to_vec());
        i = found + 1;
        k = found + 3;
    }
    match &pat[i..end] {
        [] => chunks
            .last_mut()
            .into_iter()
            .for_each(|last| last.push('-')),
        chunk => chunks.push(chunk.to_vec()),
    }
    for k in (1..chunks.len()).rev() {
        if chunks[k - 1].last() > chunks[k].first() {
            let next = chunks.remove(k);
            chunks[k - 1].pop();
            chunks[k - 1].extend_from_slice(&next[1..]);
        }
    }
    chunks
        .iter()
        .map(|chunk| escape_class_chars(chunk, true))
        .collect::<Vec<_>>()
        .join(&'-')
}

/// A `[source]` class. An empty class never matches, and a lone `!` matches any character.
fn class(source: &[char]) -> Token {
    match source {
        [] => return Token::Never,
        ['!'] => return Token::Any,
        _ => {}
    }
    let (negated, body): (bool, Vec<char>) = match source[0] {
        '!' => (true, source[1..].to_vec()),
        '^' | '[' => (false, [&['\\'], source].concat()),
        _ => (false, source.to_vec()),
    };
    let mut chars = body.into_iter();
    let mut pending: Vec<(char, bool)> = std::iter::from_fn(|| {
        let c = chars.next()?;
        Some(if c == '\\' {
            (chars.next().unwrap_or('\\'), true)
        } else {
            (c, false)
        })
    })
    .collect();
    let mut items: Vec<(char, char)> = Vec::new();
    pending.reverse();
    while let Some((lo, _)) = pending.pop() {
        match pending.last() {
            Some(&('-', false)) if pending.len() >= 2 => {
                pending.pop();
                let (hi, _) = pending.pop().unwrap_or((lo, false));
                items.push((lo, hi));
            }
            _ => items.push((lo, lo)),
        }
    }
    Token::Class { negated, items }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::star("admin-eu", "admin-*", true)]
    #[case::star_spans_slash("a/b/c", "a*c", true)]
    #[case::question("ab", "a?", true)]
    #[case::question_needs_one("a", "a?", false)]
    #[case::exact_is_case_sensitive("Admin", "admin", false)]
    #[case::class("b1", "[abc]1", true)]
    #[case::range("m", "[a-z]", true)]
    #[case::negated("x", "[!xyz]", false)]
    #[case::negated_other("q", "[!xyz]", true)]
    #[case::unclosed_bracket_literal("[a", "[a", true)]
    #[case::leading_close_bracket("]", "[]]", true)]
    #[case::trailing_hyphen_literal("-", "[a-]", true)]
    #[case::empty_range_never_matches("m", "[z-a]", false)]
    #[case::backslash_is_literal("\\", "[\\]", true)]
    #[case::caret_is_literal("^", "[^a]", true)]
    #[case::star_needs_suffix("admin", "*x", false)]
    fn glob_matches_names(#[case] name: &str, #[case] pattern: &str, #[case] expected: bool) {
        assert_eq!(glob_match(name, pattern), expected);
    }
}
