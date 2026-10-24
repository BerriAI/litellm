mod documents;
mod merge;
mod store;

pub use documents::{Document, documents};
pub use merge::{Merged, merge};
pub use store::{Files, Store};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;
    use serde_yaml_ng::Value;

    use super::*;
    use crate::Error;

    struct Memory(BTreeMap<&'static str, &'static str>);

    impl<const N: usize> From<[(&'static str, &'static str); N]> for Memory {
        fn from(files: [(&'static str, &'static str); N]) -> Self {
            Self(files.into())
        }
    }

    impl Store for Memory {
        type Key = &'static str;

        fn locate(
            &self,
            entry: &str,
            _: &&'static str,
            _: &&'static str,
        ) -> Result<&'static str, Error> {
            self.0
                .get_key_value(entry)
                .map(|(key, _)| *key)
                .ok_or_else(|| {
                    Error::document(
                        &entry,
                        std::io::Error::from(std::io::ErrorKind::NotFound).into(),
                    )
                })
        }

        fn read(&self, key: &&'static str) -> Result<String, Error> {
            Ok(self.0[key].to_owned())
        }
    }

    fn load(files: impl Into<Memory>) -> Result<Merged<&'static str>, Error> {
        documents(&files.into(), "root").map(|documents| merge(&documents))
    }

    fn yaml(text: &str) -> Value {
        serde_yaml_ng::from_str(text).unwrap()
    }

    #[rstest]
    fn reads_each_document_once_in_breadth_first_order() {
        let order = documents(
            &Memory::from([
                ("root", "include: [a, b]"),
                ("a", "include: [c, root]"),
                ("b", "include: [a]"),
                ("c", "{}"),
            ]),
            "root",
        )
        .unwrap()
        .into_iter()
        .map(|document| document.key)
        .collect::<Vec<_>>();
        assert_eq!(order, ["root", "a", "b", "c"]);
    }

    #[rstest]
    #[case::lists_concatenate("x: [1]", "x: [2]", "[1, 2]")]
    #[case::mappings_replace_whole("x: {a: 1, b: 2}", "x: {c: 3}", "{c: 3}")]
    #[case::scalar_replaces_list("x: [1]", "x: 2", "2")]
    #[case::list_replaces_scalar("x: 1", "x: [2]", "[2]")]
    fn merges_top_level_keys_like_python(
        #[case] root: &'static str,
        #[case] other: &'static str,
        #[case] expected: &str,
    ) {
        let root = Box::leak(format!("include: [other]\n{root}").into_boxed_str());
        let merged = load([("root", &*root), ("other", other)]).unwrap();
        assert_eq!(merged.document.get("x"), Some(&yaml(expected)));
        assert!(!merged.document.contains_key("include"));
    }

    #[rstest]
    fn keeps_concatenating_after_a_replace() {
        let merged = load([
            ("root", "include: [a, b, c]\nx: [1]"),
            ("a", "x: 2"),
            ("b", "x: [3]"),
            ("c", "x: [4, 5]"),
        ])
        .unwrap();
        assert_eq!(merged.document.get("x"), Some(&yaml("[3, 4, 5]")));
        assert_eq!(merged.locate("x", 0), Some((&"b", 0)));
        assert_eq!(merged.locate("x", 2), Some((&"c", 1)));
        assert_eq!(merged.locate("x", 3), None);
    }

    #[rstest]
    fn locates_list_items_and_replaced_values_in_their_documents() {
        let merged = load([
            (
                "root",
                "include: [a]\nmodel_list: [r0, r1]\ngeneral_settings: {k: 1}\nonly_root: 1",
            ),
            ("a", "model_list: [a0]\ngeneral_settings: {k: 2}"),
        ])
        .unwrap();
        assert_eq!(merged.locate("model_list", 1), Some((&"root", 1)));
        assert_eq!(merged.locate("model_list", 2), Some((&"a", 0)));
        assert_eq!(merged.locate("general_settings", 0), Some((&"a", 0)));
        assert_eq!(merged.locate("only_root", 0), Some((&"root", 0)));
        assert_eq!(merged.locate("missing", 0), None);
    }

    #[rstest]
    fn names_the_missing_include() {
        let error = load([("root", "include: [gone.yaml]")]).unwrap_err();
        assert!(
            matches!(error, Error::Document { ref key, .. } if key.contains("gone.yaml")),
            "{error:?}"
        );
    }

    #[rstest]
    #[case::bare_string("include: a.yaml")]
    #[case::number_item("include: [1]")]
    fn rejects_include_that_is_not_a_list_of_paths(#[case] root: &'static str) {
        assert!(load([("root", root)]).is_err());
    }

    #[rstest]
    #[case::declaring_directory_wins(&["nested/x.yaml", "x.yaml"], "nested/x.yaml")]
    #[case::falls_back_to_root_directory(&["x.yaml"], "x.yaml")]
    #[case::reports_declaring_path_when_neither_exists(&[], "nested/x.yaml")]
    fn files_resolve_against_declaring_then_root_directory(
        #[case] existing: &[&str],
        #[case] expected: &str,
    ) {
        let directory = tempfile::TempDir::new().unwrap();
        let root = directory.path().canonicalize().unwrap();
        std::fs::create_dir(root.join("nested")).unwrap();
        for file in ["config.yaml", "nested/first.yaml"].iter().chain(existing) {
            std::fs::write(root.join(file), "{}").unwrap();
        }
        let located = Files.locate(
            "x.yaml",
            &root.join("nested/first.yaml"),
            &root.join("config.yaml"),
        );
        match located {
            Ok(path) => assert_eq!(path, root.join(expected)),
            Err(Error::Document { key, .. }) if !existing.contains(&expected) => {
                assert!(key.contains(&*root.join(expected).to_string_lossy()))
            }
            Err(error) => panic!("{error:?}"),
        }
    }
}
