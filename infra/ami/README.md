# litellm-agent-runtime AMI

Builds the AMI that runs the per-session agent daemon on EC2. Consumed by
the EC2 VM provider in
`litellm/proxy/agent_session_endpoints/vm_providers/ec2.py`.

## Layout

```
infra/ami/
├── litellm-agent-runtime.pkr.hcl   # Packer config (HCL2)
├── scripts/
│   └── install-runtime.sh          # apt + node24 + python3.13 + uv + bun
├── files/
│   ├── litellm-agent-runtime.service  # systemd unit
│   └── daemon-stub.py              # placeholder daemon (Epic C replaces it)
└── README.md
```

## Prerequisites

- Packer 1.10+ (`packer version`)
- AWS CLI configured with the BYOC profile (default: `litellm-poc`)
- IAM permissions on the build profile: `ec2:RunInstances`, `ec2:CreateImage`,
  `ec2:RegisterImage`, `ec2:CreateTags`, `ec2:CreateSnapshot`,
  `ec2:CreateKeypair`, `ec2:DescribeImages`, `ec2:TerminateInstances`,
  `ec2:DeleteKeyPair` (the standard Packer `amazon-ebs` set)

## Build

```bash
cd infra/ami
packer init litellm-agent-runtime.pkr.hcl
packer build \
  -var "aws_profile=litellm-poc" \
  -var "region=us-west-2" \
  litellm-agent-runtime.pkr.hcl
```

Build runs on a `t3.large`. Total wall time on the BYOC PoC account: ~6–8
minutes (mostly apt + node-source). Cost per build: ~$0.50.

The output prints the new AMI ID, e.g.:
```
==> Builds finished. The artifacts of successful builds are:
--> litellm-agent-runtime.amazon-ebs.litellm-agent-runtime: AMIs were created:
us-west-2: ami-0abc1234...
```

Capture this AMI ID into `config.yaml`:

```yaml
agent_settings:
  vm_provider: ec2
  ec2:
    default_ami_id: ami-0abc1234...
    default_region: us-west-2
```

## Sharing the AMI cross-account

The PoC account builds privately. To share with a customer's BYOC account
without rebuilding:

```bash
packer build \
  -var "aws_profile=litellm-poc" \
  -var 'ami_users=["111122223333"]' \
  litellm-agent-runtime.pkr.hcl
```

Packer adds the `ami_users` to the AMI's `LaunchPermissions`.

## Boot modes

The daemon (stub or real) honours `LITELLM_AGENT_MODE` from EC2 user-data:

- `session` — cold-boot path. Daemon hits `/v1/internal/sessions/{sid}/bootstrap`,
  then heartbeats every 30s.
- `warm` — warm-pool path. Daemon idles waiting for hydrate (B2). The stub
  just heartbeats; the real implementation lands in LIT-2890.

Required user-data env (written by the EC2 provider):

```
LITELLM_SESSION_ID=...
LITELLM_TEAM_ID=...
LITELLM_AGENT_ID=...
LITELLM_BASE_URL=https://your-proxy/
LITELLM_AGENT_MODE=session
LITELLM_DAEMON_JWT=<scoped JWT minted by proxy>
```

The provider writes these to `/etc/litellm-agent/runtime.env` (mode 600) and
the systemd unit reads them via `EnvironmentFile=`.

## Replacing the daemon stub (Epic C)

The stub at `files/daemon-stub.py` is intentionally minimal. Epic C replaces
it by:

1. Bumping the Packer config to copy the real daemon entrypoint into
   `/opt/litellm-agent-runtime/daemon`
2. Re-running `packer build`
3. Updating `agent_settings.ec2.default_ami_id` in `config.yaml`

The systemd unit + boot-mode contract stays the same.

## Tags

Every resource Packer creates is tagged `LitellmManagedBy=agent-vm-provider`
so cleanup tools can find leaked builders. If a build is interrupted and
leaves an instance behind:

```bash
aws ec2 describe-instances \
  --filters Name=tag:LitellmManagedBy,Values=agent-vm-provider \
            Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].InstanceId' --output text \
  --profile litellm-poc \
  | xargs -n1 aws ec2 terminate-instances --profile litellm-poc --instance-ids
```
