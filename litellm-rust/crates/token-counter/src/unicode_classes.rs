use std::cmp::Ordering;
use std::sync::LazyLock;

use tokenizers::utils::SysRegex;

struct Ranges(Box<[(u32, u32)]>);

pub(super) struct UnicodeClasses {
    letters: Ranges,
    numbers: Ranges,
    spaces: Ranges,
}

static CLASSES: LazyLock<Option<UnicodeClasses>> = LazyLock::new(|| {
    let scalars: String = (0..=u32::from(char::MAX))
        .filter_map(char::from_u32)
        .collect();
    Some(UnicodeClasses {
        letters: Ranges::load(r"\p{L}+", &scalars)?,
        numbers: Ranges::load(r"\p{N}+", &scalars)?,
        spaces: Ranges::load(r"\s+", &scalars)?,
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

    pub(super) fn is_letter(&self, character: char) -> bool {
        self.letters.contains(character)
    }

    pub(super) fn is_number(&self, character: char) -> bool {
        self.numbers.contains(character)
    }

    pub(super) fn is_space(&self, character: char) -> bool {
        self.spaces.contains(character)
    }
}
