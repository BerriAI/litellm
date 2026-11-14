#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use strum::VariantArray;

    use crate::{Activation, BY_REFERENCE, DERIVED_ONLY, Field, FieldTier, Hook, OptOutBy};

    const CONTRACT: &str = include_str!("../contract.json");

    fn name<T: serde::Serialize>(value: T) -> Value {
        serde_json::to_value(value).unwrap()
    }

    fn sorted_names<T: serde::Serialize>(values: impl Iterator<Item = T>) -> Vec<String> {
        let mut names: Vec<String> = values
            .map(|value| name(value).as_str().unwrap().to_owned())
            .collect();
        names.sort();
        names
    }

    fn rust_table() -> Value {
        json!({
            "hooks": Hook::VARIANTS.iter().map(|hook| json!({
                "name": name(hook),
                "bit": hook.bit(),
                "request_path": hook.request_path(),
                "subscribable": hook.subscribable(),
                "writes": sorted_names(hook.writable().iter()),
                "outcomes": sorted_names(hook.outcomes().iter()),
            })).collect::<Vec<_>>(),
            "fields": Field::VARIANTS.iter().map(|field| json!({
                "name": name(field),
                "bit": field.bit(),
                "tier": name(field.tier()),
            })).collect::<Vec<_>>(),
            "tiers": FieldTier::VARIANTS.iter().map(|tier| json!({
                "name": name(tier),
                "rank": *tier as u8,
            })).collect::<Vec<_>>(),
            "derived_only": sorted_names(DERIVED_ONLY.iter()),
            "by_reference": sorted_names(BY_REFERENCE.iter()),
            "activations": Activation::VARIANTS.iter().map(name).collect::<Vec<_>>(),
            "opt_out_by": OptOutBy::VARIANTS.iter().map(name).collect::<Vec<_>>(),
        })
    }

    #[rstest::rstest]
    fn the_rust_enums_match_the_contract_python_also_checks() {
        let pinned: Value = serde_json::from_str(CONTRACT).unwrap();
        assert_eq!(rust_table(), pinned);
    }
}
