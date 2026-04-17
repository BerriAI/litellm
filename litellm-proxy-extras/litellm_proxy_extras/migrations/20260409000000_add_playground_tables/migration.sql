-- CreateTable
CREATE TABLE "LiteLLM_PlaygroundNode" (
    "node_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "ip_address" TEXT NOT NULL,
    "total_gpus" INTEGER NOT NULL DEFAULT 8,
    "gpu_type" TEXT NOT NULL DEFAULT 'H200',
    "is_playground_eligible" BOOLEAN NOT NULL DEFAULT false,
    "is_healthy" BOOLEAN NOT NULL DEFAULT true,
    "ssh_user" TEXT NOT NULL DEFAULT 'orchestrator',
    "model_path" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_PlaygroundNode_pkey" PRIMARY KEY ("node_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_PlaygroundBooking" (
    "booking_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "gpu_count" INTEGER NOT NULL,
    "preferred_node" TEXT,
    "status" TEXT NOT NULL DEFAULT 'allocated',
    "allocated_node" TEXT NOT NULL,
    "allocated_gpus" TEXT NOT NULL,
    "container_id" TEXT,
    "night_of" DATE NOT NULL,
    "is_overflow" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_PlaygroundBooking_pkey" PRIMARY KEY ("booking_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_UserSSHKey" (
    "ssh_key_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "public_key" TEXT NOT NULL,
    "fingerprint" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_UserSSHKey_pkey" PRIMARY KEY ("ssh_key_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_PlaygroundNode_ip_address_key" ON "LiteLLM_PlaygroundNode"("ip_address");

-- CreateIndex
CREATE INDEX "LiteLLM_PlaygroundBooking_night_of_idx" ON "LiteLLM_PlaygroundBooking"("night_of");

-- CreateIndex
CREATE INDEX "LiteLLM_PlaygroundBooking_user_id_idx" ON "LiteLLM_PlaygroundBooking"("user_id");

-- CreateIndex
CREATE INDEX "LiteLLM_PlaygroundBooking_status_idx" ON "LiteLLM_PlaygroundBooking"("status");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_UserSSHKey_fingerprint_key" ON "LiteLLM_UserSSHKey"("fingerprint");

-- CreateIndex
CREATE INDEX "LiteLLM_UserSSHKey_user_id_idx" ON "LiteLLM_UserSSHKey"("user_id");

-- AddForeignKey
ALTER TABLE "LiteLLM_PlaygroundBooking" ADD CONSTRAINT "LiteLLM_PlaygroundBooking_allocated_node_fkey" FOREIGN KEY ("allocated_node") REFERENCES "LiteLLM_PlaygroundNode"("ip_address") ON DELETE RESTRICT ON UPDATE CASCADE;
