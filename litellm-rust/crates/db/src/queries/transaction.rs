declare_queries! {
    set_lock_timeout(milliseconds: &str) => "../../../litellm/proxy/db/queries/transaction/set_lock_timeout.sql";
    set_statement_timeout(milliseconds: &str) => "../../../litellm/proxy/db/queries/transaction/set_statement_timeout.sql";
}
