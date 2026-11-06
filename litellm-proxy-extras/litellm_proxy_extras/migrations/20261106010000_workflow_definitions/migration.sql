SET LOCAL lock_timeout = '10s';

CREATE TABLE "LiteLLM_WorkflowDefinition" (
    "workflow_id" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "etag" TEXT NOT NULL,
    "digest" TEXT NOT NULL,
    "bundle_json" TEXT NOT NULL,
    "contributors" TEXT[],
    "published_snapshot_id" TEXT,
    "published_etag" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "LiteLLM_WorkflowDefinition_pkey" PRIMARY KEY ("workflow_id")
);

CREATE TABLE "LiteLLM_WorkflowSnapshot" (
    "snapshot_id" TEXT NOT NULL,
    "workflow_id" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "digest" TEXT NOT NULL,
    "bundle_json" TEXT NOT NULL,
    "contributors" TEXT[],
    "published_by" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_WorkflowSnapshot_pkey" PRIMARY KEY ("snapshot_id"),
    CONSTRAINT "LiteLLM_WorkflowSnapshot_workflow_id_fkey" FOREIGN KEY ("workflow_id")
        REFERENCES "LiteLLM_WorkflowDefinition"("workflow_id") ON DELETE RESTRICT ON UPDATE CASCADE
);

CREATE UNIQUE INDEX "LiteLLM_WorkflowDefinition_team_id_name_key" ON "LiteLLM_WorkflowDefinition"("team_id", "name");
CREATE INDEX "LiteLLM_WorkflowDefinition_team_id_workflow_id_idx" ON "LiteLLM_WorkflowDefinition"("team_id", "workflow_id");
CREATE INDEX "LiteLLM_WorkflowSnapshot_workflow_id_created_at_snapshot_id_idx" ON "LiteLLM_WorkflowSnapshot"("workflow_id", "created_at", "snapshot_id");
CREATE INDEX "LiteLLM_WorkflowSnapshot_team_id_snapshot_id_idx" ON "LiteLLM_WorkflowSnapshot"("team_id", "snapshot_id");

ALTER TABLE "LiteLLM_WorkflowRun"
    ADD COLUMN "workflow_id" TEXT,
    ADD COLUMN "snapshot_id" TEXT,
    ADD COLUMN "team_id" TEXT,
    ADD COLUMN "engine_owned" BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN "request_digest" TEXT,
    ADD COLUMN "idempotency_key" TEXT,
    ADD COLUMN "cancel_requested" BOOLEAN NOT NULL DEFAULT false;

CREATE UNIQUE INDEX "LiteLLM_WorkflowRun_principal_idempotency_key" ON "LiteLLM_WorkflowRun"("created_by", "team_id", "workflow_id", "idempotency_key");
CREATE INDEX "LiteLLM_WorkflowRun_workflow_id_team_id_created_at_run_id_idx" ON "LiteLLM_WorkflowRun"("workflow_id", "team_id", "created_at", "run_id");
CREATE INDEX "LiteLLM_WorkflowRun_team_id_engine_owned_status_idx" ON "LiteLLM_WorkflowRun"("team_id", "engine_owned", "status");
