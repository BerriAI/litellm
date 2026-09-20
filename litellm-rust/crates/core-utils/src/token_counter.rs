pub fn get_modified_max_tokens(
    user_max_tokens: i64,
    max_input_tokens: Option<i64>,
    max_output_tokens: Option<i64>,
    input_tokens: usize,
    buffer_perc: Option<f64>,
    buffer_num: Option<f64>,
) -> i64 {
    let Some(output) = max_output_tokens else {
        return user_max_tokens;
    };
    let buffer =
        (buffer_perc.unwrap_or(0.1) * input_tokens as f64).max(buffer_num.unwrap_or(10.0)) as i64;
    let input = (input_tokens as i64).saturating_add(buffer);
    if max_input_tokens == Some(output) {
        if input <= output && user_max_tokens.saturating_add(input) > output {
            return output - input;
        }
        return user_max_tokens;
    }
    user_max_tokens.min(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_context_includes_buffer_and_does_not_hide_overflow() {
        assert_eq!(
            get_modified_max_tokens(80, Some(100), Some(100), 40, None, None),
            50
        );
        assert_eq!(
            get_modified_max_tokens(80, Some(100), Some(100), 100, None, None),
            80
        );
        assert_eq!(
            get_modified_max_tokens(80, Some(100), Some(100), 90, None, None),
            0
        );
        assert_eq!(
            get_modified_max_tokens(80, Some(1000), Some(100), 900, None, None),
            80
        );
        assert_eq!(
            get_modified_max_tokens(180, Some(1000), Some(100), 900, None, None),
            100
        );
        assert_eq!(
            get_modified_max_tokens(180, None, None, 900, None, None),
            180
        );
    }
}
