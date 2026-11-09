-- AlterTable
ALTER TABLE "LiteLLM_WorkflowRun" ADD COLUMN     "engine_owned" BOOLEAN NOT NULL DEFAULT false;

-- CreateTable
CREATE TABLE "LiteLLM_WorkflowDefinition" (
    "workflow_id" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "etag" TEXT NOT NULL,
    "bundle_json" TEXT NOT NULL,
    "contributors" TEXT[],
    "published_snapshot_id" TEXT,

    CONSTRAINT "LiteLLM_WorkflowDefinition_pkey" PRIMARY KEY ("workflow_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_WorkflowSnapshot" (
    "snapshot_id" TEXT NOT NULL,
    "workflow_id" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "bundle_json" TEXT NOT NULL,
    "published_by" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "origin" TEXT NOT NULL DEFAULT 'publication',

    CONSTRAINT "LiteLLM_WorkflowSnapshot_pkey" PRIMARY KEY ("snapshot_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_WorkflowExecution" (
    "run_id" TEXT NOT NULL,
    "snapshot_id" TEXT NOT NULL,
    "team_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "principal" TEXT NOT NULL,
    "dispatch_key" TEXT NOT NULL,
    "request_digest" TEXT NOT NULL,
    "request_json" TEXT NOT NULL,
    "state" TEXT NOT NULL DEFAULT 'pending',
    "deadline" TIMESTAMP(3) NOT NULL,
    "lease_owner" TEXT,
    "lease_until" TIMESTAMP(3),
    "cancel_requested" BOOLEAN NOT NULL DEFAULT false,
    "checkpoints" JSONB NOT NULL DEFAULT '[]',
    "response_json" TEXT,
    "error" TEXT,

    CONSTRAINT "LiteLLM_WorkflowExecution_pkey" PRIMARY KEY ("run_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_WorkflowStep" (
    "step_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "node_id" TEXT NOT NULL,
    "state" TEXT NOT NULL DEFAULT 'running',
    "input" JSONB NOT NULL,
    "output" JSONB,
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "deadline" TIMESTAMP(3) NOT NULL,
    "started_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "finished_at" TIMESTAMP(3),
    "error" TEXT,
    "prompt_tokens" INTEGER NOT NULL DEFAULT 0,
    "completion_tokens" INTEGER NOT NULL DEFAULT 0,
    "response_cost" DOUBLE PRECISION,
    "request_id" TEXT,

    CONSTRAINT "LiteLLM_WorkflowStep_pkey" PRIMARY KEY ("step_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_WorkflowDefinition_team_id_name_key" ON "LiteLLM_WorkflowDefinition"("team_id", "name");

-- CreateIndex
CREATE INDEX "LiteLLM_WorkflowSnapshot_workflow_id_idx" ON "LiteLLM_WorkflowSnapshot"("workflow_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_WorkflowExecution_dispatch_key_key" ON "LiteLLM_WorkflowExecution"("dispatch_key");

-- CreateIndex
CREATE INDEX "LiteLLM_WorkflowExecution_team_id_name_idx" ON "LiteLLM_WorkflowExecution"("team_id", "name");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_WorkflowStep_run_id_node_id_key" ON "LiteLLM_WorkflowStep"("run_id", "node_id");
