# Migrations

`litellm-migrate` exports the `Migration` struct and the `migrate!` macro that embeds a directory of `<digits>_<description>.sql` files at compile time, sorted by numeric version

The crate does not apply or track migrations; callers decide how and when the embedded SQL runs
