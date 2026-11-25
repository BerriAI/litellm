use std::cmp::Ordering;
use std::sync::LazyLock;

use tokenizers::utils::SysRegex;

struct Ranges(Box<[(u32, u32)]>);

pub(super) struct UnicodeClasses {
    letters: Ranges,
    numbers: Ranges,
    spaces: Ranges,
    uppers: Ranges,
    lowers: Ranges,
}

static CLASSES: LazyLock<Option<UnicodeClasses>> = LazyLock::new(|| {
    let scalars: String = (0..=u32::from(char::MAX))
        .filter_map(char::from_u32)
        .collect();
    Some(UnicodeClasses {
        letters: Ranges::load(r"\p{L}+", &scalars)?,
        numbers: Ranges::load(r"\p{N}+", &scalars)?,
        spaces: Ranges::load(r"\s+", &scalars)?,
        uppers: Ranges::load(r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+", &scalars)?,
        lowers: Ranges::load(r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+", &scalars)?,
    })
});

impl Ranges {
    fn load(pattern: &str, scalars: &str) -> Option<Self> {
        let regex = SysRegex::new(pattern).ok()?;
        let ranges = regex
            .find_iter(scalars)
            .map(|(start, end)| {
                let matched = scalars.get(start..end)?;
                Some((
                    u32::from(matched.chars().next()?),
                    u32::from(matched.chars().next_back()?),
                ))
            })
            .collect::<Option<Box<[_]>>>()?;
        Some(Self(ranges))
    }

    fn contains(&self, character: char) -> bool {
        let code = u32::from(character);
        self.0
            .binary_search_by(|(low, high)| {
                if *high < code {
                    Ordering::Less
                } else if *low > code {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            })
            .is_ok()
    }
}

impl UnicodeClasses {
    pub(super) fn get() -> Option<&'static Self> {
        CLASSES.as_ref()
    }

    fn is_letter(&self, character: char) -> bool {
        self.letters.contains(character)
    }

    fn is_number(&self, character: char) -> bool {
        self.numbers.contains(character)
    }

    fn is_space(&self, character: char) -> bool {
        self.spaces.contains(character)
    }

    fn is_upper(&self, character: char) -> bool {
        self.uppers.contains(character)
    }

    fn is_lower(&self, character: char) -> bool {
        self.lowers.contains(character)
    }
}

/// `\p{L}`, `\p{N}`, `\s` and everything else, the character classes the
/// split regexes are written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Class {
    Letter,
    Number,
    Space,
    Other,
}

pub(super) fn class(character: char, unicode_classes: &UnicodeClasses) -> Class {
    match character {
        'A'..='Z' | 'a'..='z' => Class::Letter,
        '0'..='9' => Class::Number,
        '\t'..='\r' | ' ' => Class::Space,
        _ if character.is_ascii() => Class::Other,
        _ if unicode_classes.is_letter(character) => Class::Letter,
        _ if unicode_classes.is_number(character) => Class::Number,
        _ if unicode_classes.is_space(character) => Class::Space,
        _ => Class::Other,
    }
}

/// Byte length of the leading run of `run_class` characters.
pub(super) fn run_len(text: &str, run_class: Class, unicode_classes: &UnicodeClasses) -> usize {
    text.char_indices()
        .find(|(_, character)| class(*character, unicode_classes) != run_class)
        .map_or(text.len(), |(index, _)| index)
}

/// Membership in the two letter classes of the o200k split regex,
/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]` and `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`; `Lm`,
/// `Lo` and `M` are in both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Case {
    Upper,
    Lower,
    Both,
    Neither,
}

impl Case {
    pub(super) fn is_upper(self) -> bool {
        matches!(self, Case::Upper | Case::Both)
    }

    pub(super) fn is_lower(self) -> bool {
        matches!(self, Case::Lower | Case::Both)
    }
}

pub(super) fn case(character: char, unicode_classes: &UnicodeClasses) -> Case {
    match character {
        'A'..='Z' => Case::Upper,
        'a'..='z' => Case::Lower,
        _ if character.is_ascii() => Case::Neither,
        _ => match (
            unicode_classes.is_upper(character),
            unicode_classes.is_lower(character),
        ) {
            (true, true) => Case::Both,
            (true, false) => Case::Upper,
            (false, true) => Case::Lower,
            (false, false) => Case::Neither,
        },
    }
}

/// Byte length of the leading run of characters whose case passes `in_class`.
pub(super) fn case_run_len(
    text: &str,
    in_class: fn(Case) -> bool,
    unicode_classes: &UnicodeClasses,
) -> usize {
    text.char_indices()
        .find(|(_, character)| !in_class(case(*character, unicode_classes)))
        .map_or(text.len(), |(index, _)| index)
}
