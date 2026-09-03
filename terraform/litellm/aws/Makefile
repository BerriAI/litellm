# Deploy wrapper for a single-user stack. Defaults to ./dev.tfvars, which is
# gitignored: override with `make plan TFVARS=path/to/your.tfvars`, starting
# from examples/default/terraform.tfvars.example.
# The module directory is the Terraform root here because examples/default
# passes through none of the sizing knobs a trimmed config needs.
# Not named terraform.tfvars: Terraform auto-loads that for every command,
# including `terraform test`, which poisons the test suite with real values.

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
TFVARS ?= $(MAKEFILE_DIR)dev.tfvars

# The module declares no provider block, so the AWS provider resolves its region
# from the environment. Read it out of tfvars to keep one source of truth.
REGION := $(shell sed -n 's/^region *= *"\([^"]*\)".*/\1/p' "$(TFVARS)" 2>/dev/null)
export AWS_REGION := $(REGION)
export AWS_DEFAULT_REGION := $(REGION)

TF := terraform -chdir="$(MAKEFILE_DIR)"

# Lazy (=, not :=) so a missing state file does not cost a terraform call on
# every make invocation.
CLUSTER = $(shell $(TF) output -raw ecs_cluster 2>/dev/null)

.DEFAULT_GOAL := help
.PHONY: help init plan deploy destroy start stop url key dns guard-cluster guard-tfvars

# A guard target, not $(error): the latter fires at parse time and would take
# `make help` down with it on a clone that has no tfvars yet.
guard-tfvars:
	@test -s "$(TFVARS)" || { echo "no tfvars at $(TFVARS); copy examples/default/terraform.tfvars.example and pass TFVARS=<path>" >&2; exit 1; }
	@test -n "$(REGION)" || { echo "could not read \`region\` from $(TFVARS)" >&2; exit 1; }

guard-cluster: guard-tfvars
	@test -n "$(CLUSTER)" || { echo "no stack found; run 'make deploy' first" >&2; exit 1; }

help: ## Show this help
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*##"}{printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

init: guard-tfvars ## Install providers (idempotent)
	$(TF) init

plan: init ## Show what would change
	$(TF) plan -var-file="$(TFVARS)"

deploy: init ## Create or update the stack (prompts before applying)
	$(TF) apply -var-file="$(TFVARS)"

destroy: init ## Tear the whole stack down (prompts before destroying)
	$(TF) destroy -var-file="$(TFVARS)"

url: guard-tfvars ## Print the ALB URL
	@$(TF) output -raw alb_url && echo

key: guard-tfvars ## Print the LITELLM_MASTER_KEY (auto-generated if none was passed)
	@aws secretsmanager get-secret-value \
		--secret-id $$($(TF) output -raw master_key_secret_arn) \
		--query SecretString --output text

dns: guard-cluster ## CNAME DOMAIN at the ALB, e.g. make dns DOMAIN=llm.example.com ZONE=example.com
	@set -eu; \
	test -n "$(DOMAIN)" && test -n "$(ZONE)" || { \
		echo "usage: make dns DOMAIN=llm.example.com ZONE=example.com" >&2; exit 1; }; \
	alb=$$($(TF) output -raw alb_dns_name); \
	zone=$$(aws route53 list-hosted-zones-by-name --dns-name $(ZONE) \
		--query 'HostedZones[0].Id' --output text | sed 's#/hostedzone/##'); \
	aws route53 change-resource-record-sets --hosted-zone-id $$zone --change-batch \
		'{"Changes":[{"Action":"UPSERT","ResourceRecordSet":{"Name":"$(DOMAIN)","Type":"CNAME","TTL":60,"ResourceRecords":[{"Value":"'"$$alb"'"}]}}]}' \
		--query 'ChangeInfo.Status' --output text
