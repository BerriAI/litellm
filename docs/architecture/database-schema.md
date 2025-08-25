# Database Schema

```sql
-- Core authentication and authorization
CREATE TABLE "LiteLLM_VerificationToken" (
    "token" TEXT PRIMARY KEY,
    "key_name" TEXT,
    "key_alias" TEXT,
    "user_id" TEXT,
    "team_id" TEXT,
    "permissions" JSONB,
    "spend" DOUBLE PRECISION DEFAULT 0,
    "max_budget" DOUBLE PRECISION,
    "models" TEXT[],
    "metadata" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "expires" TIMESTAMP,
    FOREIGN KEY ("user_id") REFERENCES "LiteLLM_UserTable"("user_id"),
    FOREIGN KEY ("team_id") REFERENCES "LiteLLM_TeamTable"("team_id")
);

-- User management
CREATE TABLE "LiteLLM_UserTable" (
    "user_id" TEXT PRIMARY KEY,
    "user_email" TEXT UNIQUE,
    "user_role" TEXT CHECK ("user_role" IN ('admin', 'user', 'viewer')),
    "teams" TEXT[],
    "organization_id" TEXT,
    "max_budget" DOUBLE PRECISION,
    "spend" DOUBLE PRECISION DEFAULT 0,
    "user_metadata" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY ("organization_id") REFERENCES "LiteLLM_OrganizationTable"("organization_id")
);

-- Team management
CREATE TABLE "LiteLLM_TeamTable" (
    "team_id" TEXT PRIMARY KEY,
    "team_alias" TEXT,
    "organization_id" TEXT,
    "admins" TEXT[],
    "members" TEXT[],
    "max_budget" DOUBLE PRECISION,
    "spend" DOUBLE PRECISION DEFAULT 0,
    "models" TEXT[],
    "blocked" BOOLEAN DEFAULT FALSE,
    "metadata" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY ("organization_id") REFERENCES "LiteLLM_OrganizationTable"("organization_id")
);

-- Usage tracking
CREATE TABLE "LiteLLM_SpendLogs" (
    "request_id" TEXT PRIMARY KEY,
    "api_key" TEXT,
    "model" TEXT,
    "api_provider" TEXT,
    "total_tokens" INTEGER,
    "prompt_tokens" INTEGER,
    "completion_tokens" INTEGER,
    "spend" DOUBLE PRECISION,
    "user" TEXT,
    "team_id" TEXT,
    "metadata" JSONB DEFAULT '{}',
    "payload" JSONB,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_api_key (api_key),
    INDEX idx_created_at (created_at),
    INDEX idx_user (user),
    INDEX idx_team_id (team_id)
);

-- Model configuration
CREATE TABLE "LiteLLM_ProxyModelTable" (
    "model_id" TEXT PRIMARY KEY,
    "model_name" TEXT NOT NULL,
    "litellm_params" JSONB NOT NULL,
    "model_info" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name)
);
```
