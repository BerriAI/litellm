// Packer config for the `litellm-agent-runtime` AMI.
//
// Builds an Ubuntu 24.04 AMI in the BYOC AWS account containing:
//   - node 24, python 3.13, git, gh CLI, uv, bun
//   - the litellm-agent-runtime daemon stub (replaced in Epic C)
//   - systemd unit `litellm-agent-runtime.service` (autostart on boot)
//
// Two boot modes are honoured by the daemon, switched via the
// `LITELLM_AGENT_MODE` env in EC2 user-data:
//   - `session`: cold-boot, daemon hits /v1/internal/sessions/{sid}/bootstrap
//   - `warm`:    warm-pool, daemon idles waiting for hydrate (B2)
//
// Build:
//   packer init litellm-agent-runtime.pkr.hcl
//   packer build -var "aws_profile=litellm-poc" litellm-agent-runtime.pkr.hcl

packer {
  required_plugins {
    amazon = {
      source  = "github.com/hashicorp/amazon"
      version = ">= 1.3.0"
    }
  }
}

variable "aws_profile" {
  type        = string
  default     = "litellm-poc"
  description = "Local AWS CLI profile to use for the build."
}

variable "region" {
  type        = string
  default     = "us-west-2"
  description = "Region in which to build the AMI."
}

variable "instance_type" {
  type        = string
  default     = "t3.large"
  description = "Builder instance type. Stays under the t3.large safety cap."
}

variable "source_ami_filter_owner" {
  type    = string
  default = "099720109477"  // Canonical
}

variable "source_ami_filter_name" {
  type    = string
  default = "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"
}

variable "ami_name_prefix" {
  type    = string
  default = "litellm-agent-runtime"
}

variable "ami_users" {
  type        = list(string)
  default     = []
  description = "AWS account IDs to share the AMI with. Empty = private."
}

source "amazon-ebs" "litellm-agent-runtime" {
  profile       = var.aws_profile
  region        = var.region
  instance_type = var.instance_type

  ami_name        = "${var.ami_name_prefix}-{{timestamp}}"
  // AWS DescribeImage rejects non-ASCII; keep description ASCII-only.
  ami_description = "LiteLLM agent runtime - node24/py3.13/git/gh/uv/bun + daemon stub"
  ami_users       = var.ami_users

  source_ami_filter {
    filters = {
      name                = var.source_ami_filter_name
      root-device-type    = "ebs"
      virtualization-type = "hvm"
      architecture        = "x86_64"
    }
    owners      = [var.source_ami_filter_owner]
    most_recent = true
  }

  ssh_username = "ubuntu"

  // Use IMDSv2 only.
  imds_support = "v2.0"

  tags = {
    Name             = "${var.ami_name_prefix}"
    BuiltBy          = "litellm-packer"
    Source           = "litellm-agent-runtime.pkr.hcl"
    LitellmManagedBy = "agent-vm-provider"
  }

  run_tags = {
    Name             = "${var.ami_name_prefix}-builder"
    LitellmManagedBy = "agent-vm-provider"
  }
}

build {
  name = "litellm-agent-runtime"

  sources = ["source.amazon-ebs.litellm-agent-runtime"]

  // Wait for cloud-init to finish so apt isn't locked.
  provisioner "shell" {
    inline = [
      "cloud-init status --wait || true",
      "sudo mkdir -p /opt/litellm-agent-runtime",
    ]
  }

  // Apt deps + python 3.13 PPA + node 24 + uv + bun.
  // Each install pinned to current major versions; uv/bun are versioned
  // releases pulled by their official installers.
  provisioner "shell" {
    script = "scripts/install-runtime.sh"
  }

  // Drop the daemon stub + systemd unit.
  provisioner "file" {
    source      = "files/daemon-stub.py"
    destination = "/tmp/daemon-stub.py"
  }

  provisioner "file" {
    source      = "files/litellm-agent-runtime.service"
    destination = "/tmp/litellm-agent-runtime.service"
  }

  provisioner "shell" {
    inline = [
      "sudo install -m 0755 /tmp/daemon-stub.py /opt/litellm-agent-runtime/daemon",
      "sudo install -m 0644 /tmp/litellm-agent-runtime.service /etc/systemd/system/litellm-agent-runtime.service",
      "sudo systemctl daemon-reload",
      "sudo systemctl enable litellm-agent-runtime.service",
    ]
  }
}
