//! tiktoken's byte-level BPE: a rank file of `base64(token) rank` lines and
//! the merge loop that turns one regex piece into tokens. The merge order is
//! tiktoken's (lowest rank first, leftmost pair on ties) so the token count is
//! identical, but pairs are tracked in a heap so a long piece costs
//! `O(n log n)` instead of tiktoken's `O(n^2)`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rustc_hash::FxHashMap;

use crate::Error;

type Rank = u32;

const NO_RANK: Rank = Rank::MAX;
const END: usize = usize::MAX;

pub(super) struct MergeRanks(FxHashMap<Box<[u8]>, Rank>);

impl MergeRanks {
    pub(super) fn parse(text: &str) -> Result<Self, Error> {
        let ranks = text
            .lines()
            .filter(|line| !line.is_empty())
            .map(parse_line)
            .collect::<Result<FxHashMap<_, _>, _>>()?;
        if let Some(byte) = (0..=u8::MAX).find(|byte| !ranks.contains_key(&[*byte][..])) {
            return Err(Error::Ranks(format!("byte 0x{byte:02X} has no token")));
        }
        Ok(Self(ranks))
    }

    fn rank(&self, bytes: &[u8]) -> Rank {
        self.0.get(bytes).copied().unwrap_or(NO_RANK)
    }

    /// Token count of one regex piece, as `encode_ordinary` would produce.
    pub(super) fn count_piece(&self, piece: &[u8], scratch: &mut MergeScratch) -> usize {
        if piece.len() < 2 || self.0.contains_key(piece) {
            return 1;
        }
        scratch.reset(piece.len());
        for start in 0..piece.len() - 1 {
            scratch.set_rank(start, self.rank(&piece[start..start + 2]));
        }
        let mut parts = piece.len();
        while let Some(Reverse((rank, start))) = scratch.heap.pop() {
            if scratch.next[start] == END || scratch.rank[start] != rank {
                continue;
            }
            let merged = scratch.next[start];
            let after = scratch.next[merged];
            scratch.next[merged] = END;
            scratch.next[start] = after;
            parts -= 1;
            if after < piece.len() {
                scratch.prev[after] = start;
                scratch.set_rank(start, self.rank(&piece[start..scratch.end(after)]));
            } else {
                scratch.rank[start] = NO_RANK;
            }
            let before = scratch.prev[start];
            if before != END {
                scratch.set_rank(before, self.rank(&piece[before..scratch.end(start)]));
            }
        }
        parts
    }
}

fn parse_line(line: &str) -> Result<(Box<[u8]>, Rank), Error> {
    let (token, rank) = line
        .split_once(' ')
        .ok_or_else(|| Error::Ranks(format!("line without a rank: {line:?}")))?;
    let bytes = STANDARD
        .decode(token)
        .map_err(|error| Error::Ranks(format!("token is not base64: {error}")))?;
    let rank = rank
        .parse()
        .map_err(|error| Error::Ranks(format!("rank is not an integer: {error}")))?;
    Ok((bytes.into_boxed_slice(), rank))
}

/// Buffers reused across the pieces of one text. Parts are addressed by the
/// byte offset they start at, which also gives the leftmost-pair tie break.
#[derive(Default)]
pub(super) struct MergeScratch {
    next: Vec<usize>,
    prev: Vec<usize>,
    rank: Vec<Rank>,
    heap: BinaryHeap<Reverse<(Rank, usize)>>,
}

impl MergeScratch {
    fn reset(&mut self, len: usize) {
        self.next.clear();
        self.next.extend(1..=len);
        self.prev.clear();
        self.prev.push(END);
        self.prev.extend(0..len - 1);
        self.rank.clear();
        self.rank.resize(len, NO_RANK);
        self.heap.clear();
    }

    fn end(&self, start: usize) -> usize {
        self.next[start]
    }

    fn set_rank(&mut self, start: usize, rank: Rank) {
        self.rank[start] = rank;
        if rank != NO_RANK {
            self.heap.push(Reverse((rank, start)));
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    fn ranks() -> MergeRanks {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../litellm/litellm_core_utils/tokenizers/9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
        );
        MergeRanks::parse(&std::fs::read_to_string(path).expect("cl100k rank file is in the repo"))
            .expect("rank file parses")
    }

    /// tiktoken's `_byte_pair_merge`, transcribed, as the reference.
    fn reference_count(ranks: &MergeRanks, piece: &[u8]) -> usize {
        if piece.len() < 2 || ranks.0.contains_key(piece) {
            return 1;
        }
        let mut parts: Vec<(usize, Rank)> = (0..piece.len() - 1)
            .map(|index| (index, ranks.rank(&piece[index..index + 2])))
            .chain([(piece.len() - 1, NO_RANK), (piece.len(), NO_RANK)])
            .collect();
        let get_rank = |parts: &[(usize, Rank)], index: usize| {
            if index + 3 < parts.len() {
                ranks.rank(&piece[parts[index].0..parts[index + 3].0])
            } else {
                NO_RANK
            }
        };
        loop {
            let Some(index) = parts[..parts.len() - 1]
                .iter()
                .enumerate()
                .filter(|(_, (_, rank))| *rank != NO_RANK)
                .min_by_key(|(index, (_, rank))| (*rank, *index))
                .map(|(index, _)| index)
            else {
                return parts.len() - 1;
            };
            if index > 0 {
                parts[index - 1].1 = get_rank(&parts, index - 1);
            }
            parts[index].1 = get_rank(&parts, index);
            parts.remove(index + 1);
        }
    }

    #[test]
    fn every_byte_is_a_token() {
        let ranks = ranks();
        assert_eq!(ranks.0.len(), 100_256);
        assert!((0..=u8::MAX).all(|byte| ranks.rank(&[byte]) != NO_RANK));
    }

    #[test]
    fn heap_merge_matches_tiktokens_merge_loop() {
        let ranks = ranks();
        let mut scratch = MergeScratch::default();
        let mut rng = StdRng::seed_from_u64(99);
        let alphabet = b" abcdeorstn.,'\n\xc3\xa9\xe2\x82\xac0123";
        for _ in 0..20_000 {
            let piece: Vec<u8> = (0..rng.gen_range(1..24))
                .map(|_| alphabet[rng.gen_range(0..alphabet.len())])
                .collect();
            assert_eq!(
                ranks.count_piece(&piece, &mut scratch),
                reference_count(&ranks, &piece),
                "piece {:?}",
                String::from_utf8_lossy(&piece)
            );
        }
    }

    #[test]
    fn long_repeated_runs_stay_cheap() {
        let ranks = ranks();
        let mut scratch = MergeScratch::default();
        let piece = vec![b' '; 1 << 20];
        let started = std::time::Instant::now();
        let count = ranks.count_piece(&piece, &mut scratch);
        assert!(count > 0);
        assert!(started.elapsed().as_secs() < 5, "{:?}", started.elapsed());
    }

    #[test]
    fn malformed_rank_files_are_rejected() {
        assert!(MergeRanks::parse("IQ==").is_err());
        assert!(MergeRanks::parse("IQ== x").is_err());
        assert!(MergeRanks::parse("!!! 1").is_err());
        assert!(MergeRanks::parse("IQ== 1").is_err());
    }
}
