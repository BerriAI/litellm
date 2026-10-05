# Migrations

`litellm-migrate` embeds `<digits>_<description>.sql` files with checksums and computes pending migrations from caller-supplied applied versions

Applying and recording migrations is the caller's job

Migration files are append-only, and changing an applied file is an error

Recording happens after execution, so concurrent processes can both apply a pending migration. Every migration must be safe to run more than once
