WITH deleted AS (
    DELETE FROM "LiteLLM_VerificationToken"
    WHERE user_id = $1
      AND team_id = $2
    RETURNING *
), archived AS (
    INSERT INTO "LiteLLM_DeletedVerificationToken" (
        id, access_group_ids, agent_id, aliases, allowed_cache_controls, allowed_routes, auto_rotate, blocked,
        budget_duration, budget_fallbacks, budget_id, budget_reset_at, config, created_at, created_by, expires,
        key_alias, key_id, key_name, key_rotation_at, key_type, last_active, last_rotation_at, max_budget,
        max_parallel_requests, metadata, model_max_budget, model_spend, models, object_permission_id,
        organization_id, permissions, policies, project_id, rotation_count, rotation_interval, router_settings,
        rpm_limit, settings_updated_at, soft_budget_cooldown, spend, team_id, token, total_spend, tpd_limit,
        tpm_limit, updated_at, updated_by, user_id, deleted_at
    )
    SELECT gen_random_uuid()::text,
           COALESCE(access_group_ids, '{}'), agent_id, aliases, COALESCE(allowed_cache_controls, '{}'),
           COALESCE(allowed_routes, '{}'), auto_rotate, blocked, budget_duration, budget_fallbacks, budget_id,
           budget_reset_at, config, created_at, created_by, expires, key_alias, key_id, key_name, key_rotation_at,
           key_type, last_active, last_rotation_at, max_budget, max_parallel_requests, metadata, model_max_budget,
           model_spend, COALESCE(models, '{}'), object_permission_id, organization_id, permissions,
           COALESCE(policies, '{}'), project_id, rotation_count, rotation_interval, router_settings, rpm_limit,
           settings_updated_at, soft_budget_cooldown, spend, team_id, token, total_spend, tpd_limit, tpm_limit,
           updated_at, updated_by, user_id, NOW() AT TIME ZONE 'UTC'
    FROM deleted
)
SELECT d.token AS "token!",
       m.jwt_issuer AS "jwt_issuer?",
       m.jwt_claim_name AS "jwt_claim_name?",
       m.jwt_claim_value AS "jwt_claim_value?"
FROM deleted AS d
LEFT JOIN "LiteLLM_JWTKeyMapping" AS m ON m.token = d.token
ORDER BY d.token
