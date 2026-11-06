# Provider integration tests

These Go acceptance tests run Terraform against a real proxy with Postgres and an enterprise license. They are opt-in through `TF_ACC=1` and skip during the provider workflow's ordinary `go test ./...` run

From `terraform/provider`, run `TF_ACC=1 TF_ACC_TERRAFORM_PATH="$(command -v terraform)" LITELLM_API_BASE=http://127.0.0.1:4107 LITELLM_API_KEY="$LITELLM_MASTER_KEY" go test ./tests/integration -run TestAccTagModelBudgetDrift -count=1 -v`

The tag budget test ignores `budget_id` changes and omits `budget_id` from the out-of-band update to isolate Read-side drift detection from proxy issue #46999. The SDK harness creates a unique tag and destroys it after the test
