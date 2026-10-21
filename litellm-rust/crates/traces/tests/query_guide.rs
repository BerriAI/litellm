use litellm_traces::query::guide::{Example, QueryGuide, Section};
use rstest::rstest;

#[rstest]
#[case::empty(false)]
#[case::supplied(true)]
fn guide_preserves_supplied_content_and_order(#[case] populated: bool) {
    let sql = "SELECT 'quotes', '<&>', '{{ sql }}', '{% block %}'\nFROM supplied_table\nLIMIT 7";
    let sections = [
        Section {
            title: "First section",
            body: "Backend content <&> {{ untouched }}",
        },
        Section {
            title: "Second section",
            body: "Second body",
        },
    ];
    let examples = [
        Example {
            name: "First example".into(),
            sql: sql.into(),
        },
        Example {
            name: "Second example".into(),
            sql: "SELECT 2".into(),
        },
    ];
    let gotchas = ["First gotcha <&>".into(), "Second gotcha".into()];
    let guide = QueryGuide {
        sections: if populated { &sections } else { &[] },
        examples: if populated { &examples } else { &[] },
        gotchas: if populated { &gotchas } else { &[] },
    }
    .render()
    .unwrap();
    assert!(guide.starts_with("Trace SQL query guide\n\n"));
    assert!(guide.contains("POST /v1/traces/query"));
    assert!(guide.contains("GET /v1/traces/query/help"));
    if !populated {
        assert!(!guide.contains(sections[0].title));
        assert!(!guide.contains(&examples[0].name));
        assert!(!guide.contains(&gotchas[0]));
        return;
    }
    let contents = [
        sections[0].title,
        sections[0].body,
        sections[1].title,
        sections[1].body,
        "Endpoints",
        "Examples",
        &examples[0].name,
        sql,
        &examples[1].name,
        &examples[1].sql,
        "Gotchas",
        &gotchas[0],
        &gotchas[1],
    ];
    let positions = contents.map(|text| guide.find(text).expect(text));
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(guide.contains(&format!(
        "{}\n\n{}\n\n",
        sections[0].title, sections[0].body
    )));
    assert!(guide.contains(&format!("{}\n{}\n\n", examples[0].name, sql)));
}
