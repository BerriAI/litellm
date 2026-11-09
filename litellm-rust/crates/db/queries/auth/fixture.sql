INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, tpm_limit, rpm_limit, allowed_models, created_by, updated_by)
VALUES ('auth-b', 0, 22, 33, NULL, 'auth-test', 'auth-test'),
       ('auth-b-key', 4.5, 44, 55, ARRAY['key-model'], 'auth-test', 'auth-test'),
       ('auth-b-org', 13, 66, 77, ARRAY['org-model'], 'auth-test', 'auth-test');
INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id, models, mcp_servers, agents)
VALUES ('auth-op', ARRAY['allowed-model'], NULL, ARRAY['auth-agent']);
INSERT INTO "LiteLLM_OrganizationTable" (organization_id, organization_alias, budget_id, object_permission_id, models, spend, created_by, updated_by)
VALUES ('auth-org', 'Auth Org', 'auth-b-org', 'auth-op', NULL, 13.5, 'auth-test', 'auth-test');
INSERT INTO "LiteLLM_ModelTable" (id, aliases, created_by, updated_by)
VALUES (-922, '{"alias":"actual"}', 'auth-test', 'auth-test');
INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, organization_id, model_id, object_permission_id, models, admins, members, spend, max_budget, blocked, members_with_roles)
VALUES ('auth-team', 'Auth Team', 'auth-org', -922, 'auth-op', ARRAY['team-model'], NULL, NULL, 2.5, 25, true, '[{"user_id":"auth-u","role":"admin"}]'),
       ('auth-team-empty', 'Auth Empty', NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL, false, '[]');
INSERT INTO "LiteLLM_UserTable" (user_id, user_email, sso_user_id, models, teams, spend)
VALUES ('auth-u', 'Test_%@Example.com', 'auth-sso', NULL, ARRAY['auth-team'], 3.5),
       ('auth-other', 'test-xx@example.com', NULL, ARRAY['other-model'], NULL, 0);
INSERT INTO "LiteLLM_OrganizationMembership" (user_id, organization_id, user_role, budget_id)
VALUES ('auth-u', 'auth-org', 'org_admin', 'auth-b');
INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend, budget_id)
VALUES ('auth-u', 'auth-team', 6.5, 'auth-b');
INSERT INTO "LiteLLM_ProjectTable" (project_id, project_alias, team_id, budget_id, object_permission_id, models, spend, created_by, updated_by)
VALUES ('auth-project', 'Auth Project', 'auth-team', 'auth-b', 'auth-op', NULL, 8.5, 'auth-test', 'auth-test');
INSERT INTO "LiteLLM_VerificationToken" (token, user_id, team_id, organization_id, project_id, budget_id, spend, models)
VALUES (repeat('a', 64), 'auth-u', 'auth-team', 'auth-org', 'auth-project', 'auth-b-key', 4.5, ARRAY['key-model']),
       (repeat('e', 64), NULL, NULL, NULL, NULL, NULL, 0, ARRAY[]::text[]);
INSERT INTO "LiteLLM_DeprecatedVerificationToken" (id, token, active_token_id, revoke_at)
VALUES ('auth-deprecated', repeat('b', 64), repeat('a', 64), now() + INTERVAL '1 hour'),
       ('auth-expired', repeat('c', 64), repeat('a', 64), now() - INTERVAL '1 hour'),
       ('auth-chained', repeat('d', 64), repeat('b', 64), now() + INTERVAL '1 hour'),
       ('auth-shadowed', repeat('a', 64), repeat('e', 64), now() + INTERVAL '1 hour');
INSERT INTO "LiteLLM_EndUserTable" (user_id, models, budget_id, object_permission_id, spend, blocked)
VALUES ('auth-end', NULL, 'auth-b', 'auth-op', 10.5, false),
       ('auth-unrestricted', NULL, NULL, NULL, 0, false);
INSERT INTO "LiteLLM_TagTable" (tag_name, models, budget_id, spend)
VALUES ('auth-tag', NULL, 'auth-b', 11.5);
INSERT INTO "LiteLLM_ModelAccessGroupBudgetTable" (access_group_name, budget_id, spend)
VALUES ('auth-model-group', 'auth-b', 12.5);
INSERT INTO "LiteLLM_AccessGroupTable" (access_group_id, access_group_name, access_model_names, access_agent_ids, access_mcp_server_ids)
VALUES ('auth-access', 'Auth Access', ARRAY['group-model'], ARRAY['auth-agent'], NULL);
INSERT INTO "LiteLLM_AgentsTable" (agent_id, agent_name, agent_card_params, object_permission_id, identity_managed, enabled, extra_headers, agent_access_groups, created_by, updated_by)
VALUES ('auth-agent', 'Auth Agent', '{}', 'auth-op', true, false, NULL, ARRAY['auth-group'], 'auth-test', 'auth-test');
INSERT INTO "LiteLLM_ManagedVectorStoresTable" (vector_store_id, custom_llm_provider, litellm_params, team_id, updated_at)
VALUES ('auth-vector', 'openai', '{"api_base":"https://example.com"}', 'auth-team', now());
INSERT INTO "LiteLLM_AgentIdentity" (agent_id, provider, issuer, tenant_id, client_id, revision, active, required_roles, required_scopes)
VALUES ('auth-agent', 'microsoft_entra', 'auth-issuer', 'auth-tenant', 'auth-client', 'auth-revision', false, NULL, ARRAY['read']);
INSERT INTO "LiteLLM_RetiredAgentIdentity" (binding_id, provider, issuer, tenant_id, client_id)
VALUES ('auth-retired-binding', 'microsoft_entra', 'auth-issuer', 'auth-tenant', 'auth-retired-client');
INSERT INTO "LiteLLM_RetiredAgent" (original_agent_id) VALUES ('auth-retired-agent');
INSERT INTO "LiteLLM_VerifiedSubject" (subject_id, issuer, tenant_id, oid, user_id)
VALUES ('auth-subject', 'auth-issuer', 'auth-tenant', 'auth-oid', 'auth-u');
INSERT INTO "LiteLLM_BudgetWindowSpend" (entity_type, entity_id, window_duration, window_start, spend, updated_at)
VALUES ('key', repeat('a',64), '1h', '2025-01-01 01:00:00', 2.5, now()),
       ('team', 'auth-team', '1h', '2025-01-01 01:00:00', 2.5, now());
INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, api_key, team_id, "startTime", "endTime", spend)
VALUES ('auth-log-before', 'completion', repeat('a',64), 'auth-team', '2025-01-01 00:59:59', '2025-01-01 01:00:01', 100),
       ('auth-log-boundary', 'completion', repeat('a',64), 'auth-team', '2025-01-01 01:00:00', '2025-01-01 01:00:01', 2.5),
       ('auth-log-after', 'completion', repeat('a',64), 'auth-team', '2025-01-01 01:00:01', '2025-01-01 01:00:02', 3.5),
       ('auth-log-other', 'completion', repeat('e',64), 'auth-team-empty', '2025-01-01 01:00:00', '2025-01-01 01:00:01', 500);
