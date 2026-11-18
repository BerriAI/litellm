# Repositories crate

Read [README.md](README.md) before adding or changing a repository or a query. It sets the layout (one folder per repository, with `repository.rs`, `queries.rs` and `queries/*.sql`), the rule that queries stay private to their folder and run only through `Database::run`, and how to test and refresh `.sqlx`

- Keep [README.md](README.md) in step with the code. A change to the layout, `Database`, or the test and metadata workflow updates it in the same commit
- Connection concerns (pools, endpoints, error classification, `observe`, `within`, `Statements`) belong in `crates/db`, never here. See `crates/db/AGENTS.md`
