#[cfg_attr(
    not(all(test, feature = "schema")),
    expect(
        unused_macros,
        reason = "no query is declared yet; the first ported query uses it"
    )
)]
macro_rules! declare_queries {
    ($($cardinality:ident $name:ident($params:ident { $($field:ident: $type:ty),* $(,)? }) $(-> $row:ident)? => $file:literal;)+) => {
        $(
            #[cfg_attr(
                not(feature = "schema"),
                expect(dead_code, reason = "built by Python until Rust executes the query")
            )]
            #[derive(Debug, Clone, PartialEq)]
            #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
            struct $params {
                $($field: $type),*
            }

            declare_queries!(@check $name($params { $($field),* }) $(-> $row)? => $file);
        )+

        #[cfg(feature = "schema")]
        pub(crate) const QUERIES: &[$crate::codegen::QuerySpec] = &[
            $(declare_queries!(@spec $cardinality $name($params { $($field),* }) $(-> $row)? => $file)),+
        ];
    };
    (@check $name:ident($params:ident { $($field:ident),* }) -> $row:ident => $file:literal) => {
        #[expect(dead_code, reason = "declared for the compile-time check; Python executes it until Rust takes it over")]
        fn $name(params: &$params) {
            let _ = sqlx::query_file_as!($row, $file $(, params.$field)*);
        }
    };
    (@check $name:ident($params:ident { $($field:ident),* }) => $file:literal) => {
        #[expect(dead_code, reason = "declared for the compile-time check; Python executes it until Rust takes it over")]
        fn $name(params: &$params) {
            let _ = sqlx::query_file!($file $(, params.$field)*);
        }
    };
    (@spec Execute $name:ident($params:ident { $($field:ident),* }) -> $row:ident => $file:literal) => {
        compile_error!(concat!("`", stringify!($name), "` is Execute, which returns no rows; drop `-> ", stringify!($row), "`"))
    };
    (@spec $cardinality:ident $name:ident($params:ident { $($field:ident),* }) -> $row:ident => $file:literal) => {
        $crate::codegen::QuerySpec {
            name: stringify!($name),
            file: $file,
            cardinality: $crate::codegen::Cardinality::$cardinality,
            arguments: &[$(stringify!($field)),*],
            params: $crate::codegen::Model::of::<$params>(),
            row: Some($crate::codegen::Model::of::<$row>()),
        }
    };
    (@spec Execute $name:ident($params:ident { $($field:ident),* }) => $file:literal) => {
        $crate::codegen::QuerySpec {
            name: stringify!($name),
            file: $file,
            cardinality: $crate::codegen::Cardinality::Execute,
            arguments: &[$(stringify!($field)),*],
            params: $crate::codegen::Model::of::<$params>(),
            row: None,
        }
    };
}

#[cfg_attr(
    not(all(test, feature = "schema")),
    expect(
        unused_macros,
        reason = "no query is declared yet; the first ported query uses it"
    )
)]
macro_rules! declare_rows {
    ($($(#[$attr:meta])* struct $name:ident { $($field:ident: $type:ty),* $(,)? })+) => {
        $(
            #[cfg_attr(
                not(feature = "schema"),
                expect(dead_code, reason = "row of a query Python executes until Rust takes it over")
            )]
            #[derive(Debug, Clone, PartialEq)]
            #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
            $(#[$attr])*
            struct $name {
                $($field: $type),*
            }
        )+
    };
}

#[cfg(feature = "schema")]
pub const CATALOG: &[&[crate::codegen::QuerySpec]] = &[];

#[cfg(all(test, feature = "schema"))]
mod tests;
