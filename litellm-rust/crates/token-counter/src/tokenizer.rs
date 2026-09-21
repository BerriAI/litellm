use crate::Error;

pub trait Tokenizer: Send + Sync {
    fn count_tokens(&self, text: &str) -> Result<usize, Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CountableRequest, TokenCounter};

    struct Characters;

    impl Tokenizer for Characters {
        fn count_tokens(&self, text: &str) -> Result<usize, Error> {
            Ok(text.chars().count())
        }
    }

    #[test]
    fn request_accounting_works_with_an_injected_backend() {
        let counter = TokenCounter::new(Characters);
        let request =
            CountableRequest::parse(br#"{"messages":[{"role":"user","content":"hello"}]}"#)
                .unwrap();
        assert_eq!(
            counter.count_request(&request).unwrap().input_tokens,
            3 + 4 + 5 + 3
        );
    }
}
