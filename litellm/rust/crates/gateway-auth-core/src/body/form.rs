//! A form's bracket-spelled metadata fields (`metadata[spend_logs_metadata][owner]`), read into
//! the mapping they spell, as the proxy's form handlers read them with
//! `extract_nested_form_metadata`.

use super::value::{CallerValue, Params};

/// The mapping the keys under `prefix` (`metadata[`, `litellm_metadata[`) build, in the order
/// written. A key whose path runs through a value that is not a mapping sets nothing.
pub(crate) fn form_metadata(body: &Params, prefix: &str) -> Params {
    body.iter().filter(|(key, _)| key.starts_with(prefix)).fold(
        Params::default(),
        |mut metadata, (key, value)| {
            let path = key.replace(prefix, "");
            let parts: Vec<&str> = path.trim_end_matches(']').split("][").collect();
            if parts.first().is_some_and(|first| !first.is_empty()) {
                insert(&mut metadata, &parts, value.clone());
            }
            metadata
        },
    )
}

fn insert(params: &mut Params, parts: &[&str], value: CallerValue) {
    match parts {
        [] => {}
        [last] => params.set(last, value),
        [first, rest @ ..] => {
            if !params.contains(first) {
                params.set(first, CallerValue::Map(Params::default()));
            }
            if let Some(CallerValue::Map(next)) = params.get_mut(first) {
                insert(next, rest, value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn text(value: &str) -> CallerValue {
        CallerValue::Text(value.to_owned())
    }

    fn form(fields: &[(&str, CallerValue)]) -> Params {
        fields
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect()
    }

    #[rstest]
    #[case::one_level(
        form(&[("metadata[tags]", text("prod")), ("model", text("whisper-1"))]),
        form(&[("tags", text("prod"))])
    )]
    #[case::nested(
        form(&[("metadata[spend][owner]", text("a")), ("metadata[spend][team]", text("b"))]),
        form(&[("spend", CallerValue::Map(form(&[("owner", text("a")), ("team", text("b"))])))])
    )]
    #[case::the_last_spelling_of_a_key_wins_in_its_first_place(
        form(&[("metadata[a]", text("1")), ("metadata[b]", text("2")), ("metadata[a]]", text("3"))]),
        form(&[("a", text("3")), ("b", text("2"))])
    )]
    #[case::an_empty_path(form(&[("metadata[]", text("x"))]), Params::default())]
    #[case::through_a_value_that_is_not_a_mapping(
        form(&[("metadata[tags]", text("prod")), ("metadata[tags][x]", text("y"))]),
        form(&[("tags", text("prod"))])
    )]
    #[case::another_prefix(form(&[("litellm_metadata[tags]", text("prod"))]), Params::default())]
    fn bracket_spelled_fields_build_the_mapping_they_spell(
        #[case] body: Params,
        #[case] expected: Params,
    ) {
        assert_eq!(form_metadata(&body, "metadata["), expected);
    }
}
