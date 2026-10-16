variable "region" {
  type    = string
  default = "us-east-1"
}

variable "name" {
  description = "Prefix for every resource, so two rigs can live in one account."
  type        = string
  default     = "litellm-loadtest"
}

variable "ssh_public_key" {
  description = "Public key installed for ec2-user on every machine (scripts/cloud-run.sh uses SSH)."
  type        = string
}

variable "allowed_cidrs" {
  description = "Addresses allowed to SSH in and to reach the gateway port, e.g. your office IP /32."
  type        = list(string)
}

variable "image_tag" {
  description = "Tag pushed by scripts/build-images.sh to the ECR repositories this stack creates."
  type        = string
}

variable "gateway_image" {
  description = "Override the gateway image (default: this stack's ECR repo at image_tag)."
  type        = string
  default     = ""
}

variable "tools_image" {
  description = "Override the load-test tools image (default: this stack's ECR repo at image_tag)."
  type        = string
  default     = ""
}

variable "gateway_count" {
  description = "Gateway replicas. More than one puts an internal NLB in front."
  type        = number
  default     = 1
}

variable "gateway_instance_type" {
  description = "Dedicated gateway machine; size it like the customer would."
  type        = string
  default     = "c7i.4xlarge"
}

variable "mock_count" {
  type    = number
  default = 2
}

variable "mock_instance_type" {
  type    = string
  default = "c7i.2xlarge"
}

variable "loadgen_count" {
  type    = number
  default = 3
}

variable "loadgen_instance_type" {
  type    = string
  default = "c7i.2xlarge"
}

variable "chat_model" {
  description = "Model behind /v1/chat/completions; groups are named <model>-0, <model>-1, ..."
  type        = string
  default     = "gpt-6-astra"
}

variable "messages_model" {
  description = "Model behind /v1/messages; groups are named <model>-0, <model>-1, ..."
  type        = string
  default     = "claude-opus-5-5"
}

variable "bedrock_fraction" {
  description = "Share of each Opus group's deployments served through Bedrock. Raise to 0.5 once the Rust gateway shapes Bedrock invoke requests (it returns 501 today)."
  type        = number
  default     = 0
}

variable "chat_groups" {
  type    = number
  default = 10
}

variable "messages_groups" {
  type    = number
  default = 10
}

variable "deployments_per_group" {
  description = "Deployments behind each model group (10 + 10 groups x 8 = 160 deployments)."
  type        = number
  default     = 8
}

variable "chat_provider" {
  description = "Provider prefix for chat deployments: openai_like until native openai lands in the Rust gateway, then openai."
  type        = string
  default     = "openai_like"
}

variable "key_count" {
  type    = number
  default = 100000
}

variable "key_seed" {
  description = "Seed shared by every machine so the gateway, the DB seed and the generators agree on the keys."
  type        = string
  default     = "litellm-loadtest"
}

variable "master_key" {
  type      = string
  default   = "sk-loadtest-master"
  sensitive = true
}

variable "mock_faults" {
  description = "mockllm -faults spec, e.g. \"chat-0-3=429:1.0\" to test routing around a bad deployment."
  type        = string
  default     = ""
}

variable "create_postgres" {
  description = "Create an RDS Postgres for virtual keys (needed once the gateway reads keys from the DB)."
  type        = bool
  default     = false
}

variable "postgres_instance_class" {
  type    = string
  default = "db.m7g.large"
}
