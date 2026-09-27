use rstest::rstest;

use crate::codegen::Cardinality;

declare_queries! {
    Many echo(EchoParams { first: String, second: i64 }) -> EchoRow => "src/queries/echo.sql";
    Execute echo_discarded(EchoDiscardedParams { first: String, second: i64 }) => "src/queries/echo.sql";
}

declare_rows! {
    struct EchoRow {
        first: Option<String>,
        second: Option<i64>,
    }
}

#[rstest]
#[case::rows(0, "echo", Cardinality::Many, true)]
#[case::no_rows(1, "echo_discarded", Cardinality::Execute, false)]
fn declarations_keep_parameter_order_and_cardinality(
    #[case] index: usize,
    #[case] name: &str,
    #[case] cardinality: Cardinality,
    #[case] has_row: bool,
) {
    let query = &QUERIES[index];

    assert_eq!(
        (
            query.name,
            query.arguments,
            query.cardinality,
            query.row.is_some()
        ),
        (name, &["first", "second"][..], cardinality, has_row),
    );
}
