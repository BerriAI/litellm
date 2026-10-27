macro_rules_attribute::attribute_alias! {
    #[apply(db_input)] =
        #[derive(Clone, Debug, PartialEq)];

    #[apply(db_row)] =
        #[derive(Clone, Debug, PartialEq, sqlx::FromRow)];
}

mod daily_spend;
mod end_user;
mod error;
mod failure;
mod spend;
#[cfg(test)]
mod test_support;

pub use daily_spend::{DailyEndUserRow, DailyTeamRow, DailyUserRow};
pub use end_user::{EndUser, RateLimits, find_end_user};
pub use error::Error;
pub use failure::DbFailure;
pub use spend::{EntitySpend, add_key_spend};
