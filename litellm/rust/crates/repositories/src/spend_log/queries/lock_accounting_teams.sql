SELECT pg_advisory_xact_lock(hashtext(id)) IS NULL AS locked
FROM (SELECT DISTINCT unnest($1::text[]) COLLATE "C" AS id ORDER BY id) AS teams
