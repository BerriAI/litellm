SELECT pg_advisory_xact_lock(hashtext($1)) IS NULL AS "locked!"
