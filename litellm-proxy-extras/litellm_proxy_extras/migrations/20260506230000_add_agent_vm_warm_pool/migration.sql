-- Adds `LiteLLM_AgentVM` to track the lifecycle of warm-pool VMs (LIT-2890).
--
-- Each row maps to one underlying instance in the team's BYOC AWS account
-- (or other provider). The maintenance loop in
-- `litellm/proxy/agent_session_endpoints/warm_pool/manager.py` keeps
-- `state='warm'` rows topped up to `LiteLLM_AgentVMConfig.warm_pool_size`,
-- and `attach.py` flips them to `hydrating` -> `attached` on session create.
--
-- On session terminate the row is moved to `terminating` -> `terminated`
-- (NOT recycled — we destroy and rebuild each VM for cross-tenant safety).

CREATE TABLE IF NOT EXISTS "LiteLLM_AgentVM" (
    "id" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "region" TEXT,
    "state" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "pool_id" TEXT NOT NULL,
    "attached_session_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "warmed_at" TIMESTAMP(3),
    "last_hydrate_at" TIMESTAMP(3),
    "terminated_at" TIMESTAMP(3),
    "metadata" JSONB,
    CONSTRAINT "LiteLLM_AgentVM_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_AgentVM_state_pool_id_idx"
    ON "LiteLLM_AgentVM"("state", "pool_id");

CREATE INDEX IF NOT EXISTS "LiteLLM_AgentVM_team_id_state_idx"
    ON "LiteLLM_AgentVM"("team_id", "state");

CREATE INDEX IF NOT EXISTS "LiteLLM_AgentVM_attached_session_id_idx"
    ON "LiteLLM_AgentVM"("attached_session_id");
