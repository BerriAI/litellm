use rand_mt::Mt;

/// CPython's `random.Random`, draw for draw, so a seeded router picks what a seeded Python
/// router picks.
pub struct PythonRandom(Mt);

impl PythonRandom {
    /// `random.seed(seed)` for a non-negative integer seed.
    pub fn seeded(seed: u64) -> Self {
        let low = (seed & u64::from(u32::MAX)) as u32;
        let high = (seed >> 32) as u32;
        let key: &[u32] = if high == 0 { &[low] } else { &[low, high] };
        Self(Mt::new_with_key(key.iter().copied()))
    }

    pub fn from_entropy() -> Self {
        Self(Mt::new_with_key(rand::random::<[u32; 4]>()))
    }

    /// `random.random()`.
    pub fn random(&mut self) -> f64 {
        let high = f64::from(self.0.next_u32() >> 5);
        let low = f64::from(self.0.next_u32() >> 6);
        (high * 67_108_864.0 + low) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// The index `random.choice(population)` picks from a population of `length` items.
    pub fn choice(&mut self, length: usize) -> Option<usize> {
        (length > 0).then(|| self.below(length))
    }

    /// The index `random.choices(population, weights=weights)[0]` picks.
    pub fn weighted_choice(&mut self, weights: &[f64]) -> Option<usize> {
        let cumulative: Vec<f64> = weights
            .iter()
            .scan(0.0, |total, weight| {
                *total += weight;
                Some(*total)
            })
            .collect();
        let total = *cumulative.last()?;
        if total <= 0.0 || !total.is_finite() {
            return None;
        }
        let point = self.random() * total;
        let index = cumulative.partition_point(|bound| *bound <= point);
        Some(index.min(weights.len() - 1))
    }

    fn below(&mut self, bound: usize) -> usize {
        let bits = usize::BITS - bound.leading_zeros();
        loop {
            let candidate = self.bits(bits);
            if candidate < bound {
                return candidate;
            }
        }
    }

    fn bits(&mut self, count: u32) -> usize {
        match count {
            0 => 0,
            1..=32 => (self.0.next_u32() >> (32 - count)) as usize,
            _ => {
                let low = self.0.next_u32() as usize;
                let high = (self.0.next_u32() >> (64 - count)) as usize;
                (high << 32) | low
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::PythonRandom;

    // Expected values are `random.Random(seed)` draws from CPython 3.13, whose algorithm
    // (Lib/random.py `random`, `choice`, `choices`, `_randbelow_with_getrandbits`) has been
    // stable since 3.2.
    #[rstest]
    #[case::small_seed(42, [0.6394267984578837, 0.025010755222666936, 0.27502931836911926])]
    fn random_matches_cpython(#[case] seed: u64, #[case] expected: [f64; 3]) {
        let mut random = PythonRandom::seeded(seed);
        assert_eq!(
            [random.random(), random.random(), random.random()],
            expected
        );
    }

    #[rstest]
    fn wide_seed_uses_both_words() {
        assert_eq!(
            PythonRandom::seeded((1 << 40) + 5).random(),
            0.5043802970418443
        );
    }

    #[rstest]
    fn choice_matches_cpython() {
        let mut random = PythonRandom::seeded(42);
        let picks: Vec<usize> = (0..5).map(|_| random.choice(5).unwrap()).collect();
        assert_eq!(picks, [0, 0, 2, 1, 1]);
    }

    #[rstest]
    fn weighted_choice_matches_cpython() {
        let mut random = PythonRandom::seeded(7);
        let picks: Vec<usize> = (0..6)
            .map(|_| random.weighted_choice(&[1.0, 0.5, 0.25]).unwrap())
            .collect();
        assert_eq!(picks, [0, 0, 1, 0, 0, 0]);
    }

    #[rstest]
    #[case::empty(&[])]
    #[case::all_zero(&[0.0, 0.0])]
    fn weighted_choice_without_positive_total_picks_nothing(#[case] weights: &[f64]) {
        assert_eq!(PythonRandom::seeded(1).weighted_choice(weights), None);
    }
}
