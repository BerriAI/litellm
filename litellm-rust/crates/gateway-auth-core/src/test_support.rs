use litellm_config::OneOrMany;

pub fn one(value: &str) -> Option<OneOrMany> {
    Some(OneOrMany::One(value.to_owned()))
}

pub fn many(values: &[&str]) -> Option<OneOrMany> {
    Some(OneOrMany::Many(
        values.iter().map(|value| (*value).to_owned()).collect(),
    ))
}
