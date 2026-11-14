-- CreateTable
CREATE TABLE "LiteLLM_WorkflowRun" (
    "run_id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "workflow_type" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "created_by" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "input" JSONB,
    "output" JSONB,
    "metadata" JSONB,

    CONSTRAINT "LiteLLM_WorkflowRun_pkey" PRIMARY KEY ("run_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_WorkflowEvent" (
    "event_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "event_type" TEXT NOT NULL,
    "step_name" TEXT NOT NULL,
    "sequence_number" INTEGER NOT NULL,
    "data" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_WorkflowEvent_pkey" PRIMARY KEY ("event_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_WorkflowMessage" (
    "message_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "role" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "sequence_number" INTEGER NOT NULL,
    "session_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_WorkflowMessage_pkey" PRIMARY KEY ("message_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_WorkflowRun_session_id_key" ON "LiteLLM_WorkflowRun"("session_id");

-- CreateIndex
CREATE INDEX "LiteLLM_WorkflowRun_workflow_type_status_idx" ON "LiteLLM_WorkflowRun"("workflow_type", "status");

-- CreateIndex
CREATE INDEX "LiteLLM_WorkflowRun_session_id_idx" ON "LiteLLM_WorkflowRun"("session_id");

-- CreateIndex
CREATE INDEX "LiteLLM_WorkflowRun_created_at_idx" ON "LiteLLM_WorkflowRun"("created_at");

-- CreateIndex
CREATE INDEX "LiteLLM_WorkflowRun_created_by_idx" ON "LiteLLM_WorkflowRun"("created_by");

-- CreateIndex
CREATE INDEX "LiteLLM_WorkflowEvent_run_id_idx" ON "LiteLLM_WorkflowEvent"("run_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_WorkflowEvent_run_id_sequence_number_key" ON "LiteLLM_WorkflowEvent"("run_id", "sequence_number");

-- CreateIndex
CREATE INDEX "LiteLLM_WorkflowMessage_run_id_idx" ON "LiteLLM_WorkflowMessage"("run_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_WorkflowMessage_run_id_sequence_number_key" ON "LiteLLM_WorkflowMessage"("run_id", "sequence_number");

-- AddForeignKey
ALTER TABLE "LiteLLM_WorkflowEvent" ADD CONSTRAINT "LiteLLM_WorkflowEvent_run_id_fkey" FOREIGN KEY ("run_id") REFERENCES "LiteLLM_WorkflowRun"("run_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_WorkflowMessage" ADD CONSTRAINT "LiteLLM_WorkflowMessage_run_id_fkey" FOREIGN KEY ("run_id") REFERENCES "LiteLLM_WorkflowRun"("run_id") ON DELETE RESTRICT ON UPDATE CASCADE;

