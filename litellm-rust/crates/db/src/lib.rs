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
pub use spend::{
    DailySpend, DailyTable, EntitySpend, SpendLog, SpendLogRow, TeamMemberSpend,
    add_end_user_spend, add_key_spend, add_team_member_spend, add_team_spend, add_user_spend,
    begin, insert_spend_logs, lock_team, upsert_daily_end_user_spend, upsert_daily_spend,
    upsert_daily_team_spend, upsert_daily_user_spend,
};
