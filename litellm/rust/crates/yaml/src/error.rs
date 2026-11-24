use std::fmt;

use saphyr_parser::ScanError;

use crate::compose::Tag;

/// Where a node starts, as PyYAML prints it: both numbers 1-based.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    pub line: usize,
    pub column: usize,
}

impl fmt::Display for Location {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "line {}, column {}", self.line, self.column)
    }
}

/// Why `yaml.safe_load` would raise, or why this port refuses what it cannot represent.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid YAML syntax")]
    Syntax(#[from] ScanError),
    #[error("expected a single document in the stream, but found another at {at}")]
    MultipleDocuments { at: Location },
    #[error("found an alias to its own enclosing node at {at}, which a plain value cannot hold")]
    RecursiveAlias { at: Location },
    #[error("nesting is deeper than {limit} levels at {at}")]
    TooDeep { limit: usize, at: Location },
    #[error("aliases expand to more than {limit} nodes")]
    TooLarge { limit: usize },
    #[error("unsupported tag {tag} at {at}")]
    UnsupportedTag { tag: Tag, at: Location },
    #[error("expected a {expected} node for {tag} at {at}")]
    WrongNode {
        tag: Tag,
        expected: &'static str,
        at: Location,
    },
    #[error("invalid {tag} value {text:?} at {at}")]
    InvalidScalar {
        tag: Tag,
        text: String,
        at: Location,
    },
    #[error("integer {text:?} does not fit in 64 bits at {at}")]
    IntOutOfRange { text: String, at: Location },
    #[error("found an unhashable mapping key at {at}")]
    UnhashableKey { at: Location },
    #[error("expected a mapping or list of mappings for merging at {at}")]
    MergeNotMapping { at: Location },
}
