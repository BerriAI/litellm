//! `format(value, f".{precision}f")` for a `float`.

fn round_half_even(digits: &str, precision: usize) -> String {
    let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
    let mut kept: Vec<u8> = whole
        .bytes()
        .chain(
            fraction
                .bytes()
                .chain(std::iter::repeat(b'0'))
                .take(precision),
        )
        .collect();
    let rest = fraction.get(precision..).unwrap_or("");
    let first = rest.bytes().next().unwrap_or(b'0');
    let tail_nonzero = rest.bytes().skip(1).any(|digit| digit != b'0');
    let last_odd = kept.last().is_some_and(|digit| (digit - b'0') % 2 == 1);
    if first > b'5' || (first == b'5' && (tail_nonzero || last_odd)) {
        let mut index = kept.len();
        loop {
            if index == 0 {
                kept.insert(0, b'1');
                break;
            }
            index -= 1;
            if kept[index] == b'9' {
                kept[index] = b'0';
            } else {
                kept[index] += 1;
                break;
            }
        }
    }
    let split = kept.len() - precision;
    let (head, tail) = kept.split_at(split);
    let head = String::from_utf8_lossy(head);
    if precision == 0 {
        head.into_owned()
    } else {
        format!("{head}.{}", String::from_utf8_lossy(tail))
    }
}

/// `format(value, f".{precision}f")`: the exact binary value rounded half to even at
/// `precision` digits, which is what CPython's `float.__format__` prints.
pub fn fixed(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    let exact = format!("{:.1100}", value.abs());
    let rounded = round_half_even(&exact, precision);
    let negative = value.is_sign_negative();
    if negative {
        format!("-{rounded}")
    } else {
        rounded
    }
}
