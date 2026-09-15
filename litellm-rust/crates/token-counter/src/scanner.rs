//! Exact token counting for tiktoken encodings. A hand-written scanner
//! reproduces the piece boundaries of the encoding's split regex, and each
//! piece is merged with the rank file. Special tokens are ordinary text, as
//! with `encode(text, disallowed_special=())`.

use std::iter;

use super::tiktoken::{MergeRanks, MergeScratch};
use super::unicode_classes::{Class, UnicodeClasses, class};
use super::{cl100k, o200k};
use crate::Error;

const MAX_DIGITS_PER_PIECE: usize = 3;

/// Byte length of the piece the split regex matches at the start of the
/// text, given the text's first character.
pub(super) type PieceLen = fn(&str, char, &UnicodeClasses) -> usize;

#[derive(Clone, Copy, Debug)]
pub(super) enum SplitPattern {
    Cl100k,
    O200k,
}

impl SplitPattern {
    fn piece_len(self) -> PieceLen {
        match self {
            Self::Cl100k => cl100k::piece_len,
            Self::O200k => o200k::piece_len,
        }
    }
}

pub(super) struct TiktokenCounter {
    ranks: MergeRanks,
    piece_len: PieceLen,
    unicode_classes: &'static UnicodeClasses,
}

impl TiktokenCounter {
    pub(super) fn from_ranks(split: SplitPattern, rank_file: &str) -> Result<Self, Error> {
        Ok(Self {
            ranks: MergeRanks::parse(rank_file)?,
            piece_len: split.piece_len(),
            unicode_classes: UnicodeClasses::get().ok_or(Error::UnicodeClasses)?,
        })
    }

    pub(super) fn count(&self, text: &str) -> usize {
        let mut scratch = MergeScratch::default();
        pieces(text, self.piece_len, self.unicode_classes)
            .map(|piece| self.ranks.count_piece(piece.as_bytes(), &mut scratch))
            .sum()
    }
}

/// The regex matches every character, so the pieces tile the text.
pub(super) fn pieces<'a>(
    text: &'a str,
    piece_len: PieceLen,
    unicode_classes: &'static UnicodeClasses,
) -> impl Iterator<Item = &'a str> {
    iter::successors(
        split_piece(text, piece_len, unicode_classes),
        move |(_, rest)| split_piece(rest, piece_len, unicode_classes),
    )
    .map(|(piece, _)| piece)
}

fn split_piece<'a>(
    text: &'a str,
    piece_len: PieceLen,
    unicode_classes: &UnicodeClasses,
) -> Option<(&'a str, &'a str)> {
    let first = text.chars().next()?;
    Some(text.split_at(piece_len(text, first, unicode_classes)))
}

/// `'(?i:s|t|re|ve|m|ll|d)`, the contraction both encodings spell out. Simple
/// case folding also maps U+017F (long s) onto `s`.
pub(super) fn contraction_len(text: &str) -> Option<usize> {
    let mut characters = text.chars();
    if characters.next()? != '\'' {
        return None;
    }
    let first = characters.next()?;
    let len = match first {
        's' | 'S' | '\u{17F}' | 'd' | 'D' | 'm' | 'M' | 't' | 'T' => first.len_utf8(),
        'l' | 'L' => matches!(characters.next(), Some('l' | 'L')).then_some(2)?,
        'v' | 'V' | 'r' | 'R' => matches!(characters.next(), Some('e' | 'E')).then_some(2)?,
        _ => return None,
    };
    Some(1 + len)
}

pub(super) fn is_newline(character: char) -> bool {
    matches!(character, '\r' | '\n')
}

/// `\p{N}{1,3}`
pub(super) fn digit_run_len(text: &str, unicode_classes: &UnicodeClasses) -> usize {
    text.chars()
        .take(MAX_DIGITS_PER_PIECE)
        .take_while(|character| class(*character, unicode_classes) == Class::Number)
        .map(char::len_utf8)
        .sum()
}
