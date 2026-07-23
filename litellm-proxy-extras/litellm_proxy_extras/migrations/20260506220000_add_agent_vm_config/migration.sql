-- Adds `LiteLLM_AgentVMConfig` for BYOC AWS settings keyed by team.
-- Used by `litellm/proxy/agent_session_endpoints/vm_providers/ec2.py` to
-- provision per-session EC2 instances in the customer's AWS account.
--
-- The `aws_creds_enc` column is a JSON blob whose individual fields are
-- encrypted via `encrypt_value_helper` (libsodium SecretBox), so partial DB
-- exports never expose the secret. Schema fields match
-- `litellm/proxy/agent_session_endpoints/vm_providers/base.Ec2Config`.

CREATE TABLE IF NOT EXISTS "LiteLLM_AgentVMConfig" (
    "team_id" TEXT NOT NULL,
    "aws_creds_enc" TEXT,
    "region" TEXT,
    "subnet_id" TEXT,
    "security_group_id" TEXT,
    "iam_instance_profile" TEXT,
    "ami_id" TEXT,
    "instance_type" TEXT,
    "use_spot" BOOLEAN NOT NULL DEFAULT true,
    "warm_pool_size" INTEGER NOT NULL DEFAULT 0,
    "network_access" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "updated_by" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_AgentVMConfig_pkey" PRIMARY KEY ("team_id")
);
