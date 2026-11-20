#![forbid(unsafe_code)]

mod parse;
mod pattern;

pub use pattern::{Pattern, fnmatchcase};
