macro_rules! declare_queries {
    ($($name:ident($($param:ident: $type:ty),* $(,)?) $(-> $row:ident)? => $file:tt;)+) => {
        $(declare_query!($name($($param: $type),*) $(-> $row)? => $file);)+
    };
}

macro_rules! declare_query {
    ($name:ident($($param:ident: $type:ty),*) -> $row:ident => $file:tt) => {
        #[expect(
            dead_code,
            reason = "declared for the compile-time check; Python executes it until Rust takes it over"
        )]
        fn $name($($param: $type),*) {
            let _ = sqlx::query_file_as!($row, $file $(, $param)*);
        }
    };
    ($name:ident($($param:ident: $type:ty),*) => $file:tt) => {
        #[expect(
            dead_code,
            reason = "declared for the compile-time check; Python executes it until Rust takes it over"
        )]
        fn $name($($param: $type),*) {
            let _ = sqlx::query_file!($file $(, $param)*);
        }
    };
}

macro_rules! declare_rows {
    ($($(#[$attr:meta])* struct $name:ident { $($field:ident: $type:ty),* $(,)? })+) => {
        $(
            #[expect(dead_code, reason = "row of a query Python executes until Rust takes it over")]
            #[derive(Debug, Clone, PartialEq)]
            #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
            $(#[$attr])*
            struct $name {
                $($field: $type),*
            }
        )+
    };
}

mod autorouter;
mod keys;
mod transaction;
