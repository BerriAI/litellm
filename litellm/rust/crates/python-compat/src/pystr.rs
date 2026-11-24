//! `str` operations that count code points the way Python does, where Rust counts bytes.

/// `len(text)`: the number of code points.
pub fn len(text: &str) -> usize {
    text.chars().count()
}

/// `text[start:end]` with Python's slice rules: negative indices count from the end, and
/// out-of-range bounds clamp instead of failing.
pub fn slice(text: &str, start: Option<i64>, end: Option<i64>) -> String {
    let characters: Vec<char> = text.chars().collect();
    let length = i64::try_from(characters.len()).unwrap_or(i64::MAX);
    let from = clamp_index(start.unwrap_or(0), length);
    let to = clamp_index(end.unwrap_or(length), length);
    if from >= to {
        return String::new();
    }
    characters[to_usize(from)..to_usize(to)].iter().collect()
}

fn clamp_index(index: i64, length: i64) -> i64 {
    if index < 0 {
        (index + length).max(0)
    } else {
        index.min(length)
    }
}

fn to_usize(index: i64) -> usize {
    usize::try_from(index).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_count_code_points_and_clamp() {
        assert_eq!(len("héllo"), 5);
        assert_eq!(slice("héllo", Some(1), Some(-1)), "éll");
        assert_eq!(slice("héllo", Some(-2), None), "lo");
        assert_eq!(slice("héllo", Some(10), None), "");
        assert_eq!(slice("héllo", None, Some(-10)), "");
        assert_eq!(slice("héllo", Some(-10), Some(2)), "hé");
    }
}
