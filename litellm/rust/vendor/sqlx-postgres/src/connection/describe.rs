use crate::error::Error;
use crate::io::StatementId;
use crate::query_as::query_as;
use crate::statement::PgStatementMetadata;
use crate::types::Json;
use crate::PgConnection;
use smallvec::SmallVec;
use sqlx_core::query_builder::QueryBuilder;
use sqlx_core::sql_str::AssertSqlSafe;

impl PgConnection {
    /// Check whether EXPLAIN statements are supported by the current connection
    fn is_explain_available(&self) -> bool {
        let parameter_statuses = &self.inner.stream.parameter_statuses;
        let is_cockroachdb = parameter_statuses.contains_key("crdb_version");
        let is_materialize = parameter_statuses.contains_key("mz_version");
        let is_questdb = parameter_statuses.contains_key("questdb_version");
        !is_cockroachdb && !is_materialize && !is_questdb
    }

    pub(crate) async fn get_nullable_for_columns(
        &mut self,
        stmt_id: StatementId,
        meta: &PgStatementMetadata,
    ) -> Result<Vec<Option<bool>>, Error> {
        if meta.columns.is_empty() {
            return Ok(vec![]);
        }

        if meta.columns.len() * 3 > 65535 {
            tracing::debug!(
                ?stmt_id,
                num_columns = meta.columns.len(),
                "number of columns in query is too large to pull nullability for"
            );
        }

        // Query for NOT NULL constraints for each column in the query.
        //
        // This will include columns that don't have a `relation_id` (are not from a table);
        // assuming those are a minority of columns, it's less code to _not_ work around it
        // and just let Postgres return `NULL`.
        //
        // Use `UNION ALL` syntax instead of `VALUES` due to frequent lack of
        // support for `VALUES` in pgwire supported databases.
        let mut nullable_query = QueryBuilder::new("SELECT NOT attnotnull FROM ( ");
        let mut separated = nullable_query.separated("UNION ALL ");

        let mut column_iter = meta.columns.iter().zip(0i32..);
        if let Some((column, i)) = column_iter.next() {
            separated.push("( SELECT ");
            separated
                .push_bind_unseparated(i)
                .push_unseparated("::int4 AS idx, ");
            separated
                .push_bind_unseparated(column.relation_id)
                .push_unseparated("::int4 AS table_id, ");
            separated
                .push_bind_unseparated(column.relation_attribute_no)
                .push_unseparated("::int2 AS col_idx ) ");
        }

        for (column, i) in column_iter {
            separated.push("( SELECT ");
            separated
                .push_bind_unseparated(i)
                .push_unseparated("::int4, ");
            separated
                .push_bind_unseparated(column.relation_id)
                .push_unseparated("::int4, ");
            separated
                .push_bind_unseparated(column.relation_attribute_no)
                .push_unseparated("::int2 ) ");
        }

        nullable_query.push(
            ") AS col LEFT JOIN pg_catalog.pg_attribute \
                ON table_id IS NOT NULL \
               AND attrelid = table_id \
               AND attnum = col_idx \
            ORDER BY idx",
        );

        let mut nullables: Vec<Option<bool>> = nullable_query
            .build_query_scalar()
            .fetch_all(&mut *self)
            .await
            .map_err(|e| {
                err_protocol!(
                    "error from nullables query: {e}; query: {:?}",
                    nullable_query.sql()
                )
            })?;

        // If the server doesn't support EXPLAIN statements, skip this step (#1248).
        if self.is_explain_available() {
            // patch up our null inference with data from EXPLAIN
            let nullable_patch = self
                .nullables_from_explain(stmt_id, meta.parameters.len())
                .await?;

            for (nullable, patch) in nullables.iter_mut().zip(nullable_patch) {
                *nullable = patch.or(*nullable);
            }
        }

        Ok(nullables)
    }

    /// Infer nullability for columns of this statement using EXPLAIN VERBOSE.
    ///
    /// This currently only marks columns that are on the inner half of an outer join
    /// and returns `None` for all others.
    async fn nullables_from_explain(
        &mut self,
        stmt_id: StatementId,
        params_len: usize,
    ) -> Result<Vec<Option<bool>>, Error> {
        let stmt_id_display = stmt_id
            .display()
            .ok_or_else(|| err_protocol!("cannot EXPLAIN unnamed statement: {stmt_id:?}"))?;

        let mut explain = format!("EXPLAIN (VERBOSE, FORMAT JSON) EXECUTE {stmt_id_display}");
        let mut comma = false;

        if params_len > 0 {
            explain += "(";

            // fill the arguments list with NULL, which should theoretically be valid
            for _ in 0..params_len {
                if comma {
                    explain += ", ";
                }

                explain += "NULL";
                comma = true;
            }

            explain += ")";
        }

        let (Json(explains),): (Json<SmallVec<[Explain; 1]>>,) =
            query_as(AssertSqlSafe(explain)).fetch_one(self).await?;

        let mut nullables = Vec::new();

        if let Some(Explain::Plan {
            plan:
                plan @ Plan {
                    output: Some(ref outputs),
                    ..
                },
        }) = explains.first()
        {
            nullables.resize(outputs.len(), None);
            visit_plan(plan, outputs, &mut nullables);
        }

        Ok(nullables)
    }
}

fn visit_plan(plan: &Plan, outputs: &[String], nullables: &mut Vec<Option<bool>>) {
    if let Some(plan_outputs) = &plan.output {
        // all outputs of a Full Join must be marked nullable
        // otherwise, all outputs of the inner half of an outer join must be marked nullable
        if plan.join_type.as_deref() == Some("Full")
            || plan.parent_relation.as_deref() == Some("Inner")
        {
            for output in plan_outputs {
                if let Some(i) = outputs.iter().position(|o| o == output) {
                    // N.B. this may produce false positives but those don't cause runtime errors
                    nullables[i] = Some(true);
                }
            }
        }
    }

    if let Some(plans) = &plan.plans {
        if let Some("Left") | Some("Right") = plan.join_type.as_deref() {
            for plan in plans {
                visit_plan(plan, outputs, nullables);
            }
        }
    }
}

#[derive(serde::Deserialize, Debug)]
#[serde(untagged)]
enum Explain {
    // NOTE: the returned JSON may not contain a `plan` field, for example, with `CALL` statements:
    // https://github.com/launchbadge/sqlx/issues/1449
    //
    // In this case, we should just fall back to assuming all is nullable.
    //
    // It may also contain additional fields we don't care about, which should not break parsing:
    // https://github.com/launchbadge/sqlx/issues/2587
    // https://github.com/launchbadge/sqlx/issues/2622
    Plan {
        #[serde(rename = "Plan")]
        plan: Plan,
    },

    // This ensures that parsing never technically fails.
    //
    // We don't want to specifically expect `"Utility Statement"` because there might be other cases
    // and we don't care unless it contains a query plan anyway.
    Other(serde::de::IgnoredAny),
}

#[derive(serde::Deserialize, Debug)]
struct Plan {
    #[serde(rename = "Join Type")]
    join_type: Option<String>,
    #[serde(rename = "Parent Relationship")]
    parent_relation: Option<String>,
    #[serde(rename = "Output")]
    output: Option<Vec<String>>,
    #[serde(rename = "Plans")]
    plans: Option<Vec<Plan>>,
}

#[test]
fn explain_parsing() {
    let normal_plan = r#"[
   {
     "Plan": {
       "Node Type": "Result",
       "Parallel Aware": false,
       "Async Capable": false,
       "Startup Cost": 0.00,
       "Total Cost": 0.01,
       "Plan Rows": 1,
       "Plan Width": 4,
       "Output": ["1"]
     }
   }
]"#;

    // https://github.com/launchbadge/sqlx/issues/2622
    let extra_field = r#"[
   {                                        
     "Plan": {                              
       "Node Type": "Result",               
       "Parallel Aware": false,             
       "Async Capable": false,              
       "Startup Cost": 0.00,                
       "Total Cost": 0.01,                  
       "Plan Rows": 1,                      
       "Plan Width": 4,                     
       "Output": ["1"]                      
     },                                     
     "Query Identifier": 1147616880456321454
   }                                        
]"#;

    // https://github.com/launchbadge/sqlx/issues/1449
    let utility_statement = r#"["Utility Statement"]"#;

    let normal_plan_parsed = serde_json::from_str::<[Explain; 1]>(normal_plan).unwrap();
    let extra_field_parsed = serde_json::from_str::<[Explain; 1]>(extra_field).unwrap();
    let utility_statement_parsed = serde_json::from_str::<[Explain; 1]>(utility_statement).unwrap();

    assert!(
        matches!(normal_plan_parsed, [Explain::Plan { plan: Plan { .. } }]),
        "unexpected parse from {normal_plan:?}: {normal_plan_parsed:?}"
    );

    assert!(
        matches!(extra_field_parsed, [Explain::Plan { plan: Plan { .. } }]),
        "unexpected parse from {extra_field:?}: {extra_field_parsed:?}"
    );

    assert!(
        matches!(utility_statement_parsed, [Explain::Other(_)]),
        "unexpected parse from {utility_statement:?}: {utility_statement_parsed:?}"
    )
}
