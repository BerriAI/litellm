-- Make sandbox template repo_url and default_branch optional.
-- Templates can now describe a runtime image without pinning a git repository
-- (the harness clones a repo only when REPO_URL is set in the container env).

ALTER TABLE "LiteLLM_ManagedAgentSandboxTemplateTable"
  ALTER COLUMN "repo_url" DROP NOT NULL,
  ALTER COLUMN "default_branch" DROP NOT NULL,
  ALTER COLUMN "default_branch" DROP DEFAULT;
