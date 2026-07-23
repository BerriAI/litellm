# EC2 provisioning spike (LIT-2888)

Spike for Epic B (Cursor SDK on LiteLLM, full-VM provisioning). Proves end-to-end
that LiteLLM can launch an EC2 in the *customer's* AWS account, push a hydrate
command to the running instance, and tear it down — all without holding AWS
credentials server-side beyond what BYOC config supplies.

**This is throwaway code.** It exists to produce one number: warm-attach latency.
That number decides B2's hydrate transport design.

## Files

- `ec2_provision_poc.py` — main spike. Launches one t3.large, fires user-data
  callback, sends an SSM RunCommand, terminates. Always cleans up via `finally`.
- `callback_receiver.py` — 5-line Flask app on port 3333. Logs every callback
  and exposes them at `/callbacks` for the spike script to poll.

## What you need before running

A one-time AWS setup (B0 already did this for the LiteLLM PoC account):

| Resource | Value (us-west-2) |
| --- | --- |
| AMI | `ami-06c6960215cdac78d` (Ubuntu 24.04) |
| Subnet | `subnet-071fed49c5887c37d` (default VPC, public, auto-public-IP) |
| Security group | `sg-01fcce386c017c9e7` (egress all, ingress none) |
| IAM instance profile | `litellm-ec2-poc` (attached `AmazonSSMManagedInstanceCore`) |

All resources are tagged `litellm-spike=b0` so they're easy to find / clean up
later.

## How to run

```bash
# Terminal 1 — Flask receiver
uv run python infra/spikes/callback_receiver.py

# Terminal 2 — ngrok tunnel
ngrok http 3333
# Copy the https://xxx.ngrok-free.app URL.

# Terminal 3 — the spike
export CALLBACK_URL="https://xxx.ngrok-free.app/spike"
uv run python infra/spikes/ec2_provision_poc.py

# Or 5 back-to-back for the consistency validation:
for i in 1 2 3 4 5; do
  uv run python infra/spikes/ec2_provision_poc.py || exit 1
done
```

The script prints a `SUMMARY:{...}` JSON line on stdout at the end of each run.
Pipe through `jq` to collect timings across runs.

## Safety

Every `RunInstances` is paired with `TerminateInstances` in a `finally` block.
If termination fails, the script attempts a tag-scoped emergency cleanup
(everything tagged `litellm-spike=<this-run-id>`).

If a script run is killed mid-way (Ctrl-C, SIGKILL), confirm no leftovers with:

```bash
aws ec2 describe-instances --profile litellm-poc --region us-west-2 \
  --filters Name=tag:litellm-spike,Values=* \
            Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,Tags]' --output table
```

## Cost

t3.large in us-west-2 = $0.0832/hr. One spike cycle ≈ 4–5 min wall clock, so
~$0.007 per run. Budget for the full spike (5 runs + 2 failure-mode tests) is
under $0.10 — well below the $1 hard cap.

## Tearing down the one-time infra (after Epic B is done)

```bash
aws iam remove-role-from-instance-profile --profile litellm-poc \
  --instance-profile-name litellm-ec2-poc --role-name litellm-ec2-poc
aws iam delete-instance-profile --profile litellm-poc \
  --instance-profile-name litellm-ec2-poc
aws iam detach-role-policy --profile litellm-poc \
  --role-name litellm-ec2-poc \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam delete-role --profile litellm-poc --role-name litellm-ec2-poc
aws ec2 delete-security-group --profile litellm-poc --region us-west-2 \
  --group-id sg-01fcce386c017c9e7
```
