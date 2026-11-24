use litellm_config::RouteList;
use rstest::rstest;

#[rstest]
#[case::list("[/a, /b/*]", RouteList::Valid(["/a".to_owned(), "/b/*".to_owned()].into()))]
#[case::empty("[]", RouteList::Valid([].into()))]
#[case::bare_string("/a/*", RouteList::Malformed)]
#[case::number("5", RouteList::Malformed)]
#[case::mapping("{a: 1}", RouteList::Malformed)]
#[case::non_string_entry("[/a, 5, /b/*]", RouteList::Malformed)]
#[case::null_entry("[/a, null]", RouteList::Malformed)]
fn route_lists_read(#[case] yaml: &str, #[case] expected: RouteList) {
    assert_eq!(
        serde_yaml_ng::from_str::<RouteList>(yaml).unwrap(),
        expected
    );
}
