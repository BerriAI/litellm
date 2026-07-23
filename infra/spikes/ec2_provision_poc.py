"""
EC2 provisioning spike (LIT-2888).

Proves end-to-end that we can:
1. Launch a t3.large EC2 with user-data that calls back to a public HTTPS URL.
2. Send a "warm-attach" command via SSM RunCommand to a *running* instance and
   measure round-trip latency. This number decides B2's hydrate transport
   design (SSM push vs long-poll daemon vs reconsider warm pool).
3. Terminate cleanly (always — even on error — via a finally block).

NOT in scope: AMI build, daemon, AgentVMProvider, LiteLLM integration.

Usage:
    # 1) Run the Flask receiver in another shell:
    #    uv run python infra/spikes/callback_receiver.py
    # 2) Run ngrok in another shell:
    #    ngrok http 3333
    # 3) Set CALLBACK_URL env var to the ngrok https URL + /spike, e.g.:
    #    export CALLBACK_URL="https://xxx.ngrok-free.app/spike"
    # 4) Run this script:
    #    uv run python infra/spikes/ec2_provision_poc.py
    # Or run 5 times back to back:
    #    for i in 1 2 3 4 5; do uv run python infra/spikes/ec2_provision_poc.py || exit 1; done

Cost: ~$0.02 per cycle (t3.large at $0.0832/hr * ~5 min).
"""

import json
import logging
import os
import sys
import time
import uuid

import boto3

# Quiet boto3 stream logging so we never accidentally surface AWS creds in logs.
boto3.set_stream_logger("botocore", logging.WARNING)

# ---- Configuration --------------------------------------------------------
# Resolved via the orchestrator's pre-flight + B0's one-time infra setup.
AWS_PROFILE = os.environ.get("AWS_PROFILE", "litellm-poc")
REGION = os.environ.get("AWS_REGION", "us-west-2")
SUBNET_ID = os.environ.get("SUBNET_ID", "subnet-071fed49c5887c37d")
SG_ID = os.environ.get("SG_ID", "sg-01fcce386c017c9e7")
IAM_PROFILE = os.environ.get("IAM_PROFILE", "litellm-ec2-poc")
AMI_ID = os.environ.get("AMI_ID", "ami-06c6960215cdac78d")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "t3.large")

CALLBACK_URL = os.environ.get("CALLBACK_URL")
if not CALLBACK_URL:
    print(
        "ERROR: CALLBACK_URL must be set (e.g. https://xxx.ngrok-free.app/spike)",
        file=sys.stderr,
    )
    sys.exit(2)

SPIKE_ID = str(uuid.uuid4())

USER_DATA = f"""#!/bin/bash
set -e
exec > /var/log/user-data.log 2>&1
echo "$(date) starting user-data"
apt-get update -y && apt-get install -y curl jq
curl -fsS -X POST "{CALLBACK_URL}" \\
  -H 'Content-Type: application/json' \\
  -d '{{"phase":"boot","spike_id":"{SPIKE_ID}","hostname":"'"$(hostname)"'","ts":"'"$(date -u +%FT%TZ)"'"}}'
echo "$(date) callback sent"
"""


def _wait_for_callback(
    callback_base: str, spike_id: str, phase: str, timeout_s: float = 120.0
) -> float | None:
    """Poll the receiver's /callbacks endpoint until we see (spike_id, phase)
    or timeout. Returns the wall-clock seconds to first sighting (or None on
    timeout). The receiver is a sibling process; we hit its public URL via
    ngrok the same way EC2 does.

    callback_base: the ngrok URL minus /spike, e.g. https://xxx.ngrok-free.app
    """
    import urllib.request

    list_url = callback_base.rstrip("/") + "/callbacks"
    started = time.time()
    deadline = started + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(list_url, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            for entry in payload.get("callbacks", []):
                body = entry.get("body") or {}
                if body.get("spike_id") == spike_id and body.get("phase") == phase:
                    return time.time() - started
        except Exception:
            pass
        time.sleep(0.5)
    return None


def main() -> int:
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
    ec2 = session.client("ec2")
    ssm = session.client("ssm")

    callback_base = CALLBACK_URL.rsplit("/", 1)[0]

    instance_id: str | None = None
    measurements: dict[str, float | None] = {
        "boot_to_running_s": None,
        "running_to_user_data_done_s": None,
        "warm_attach_ssm_ms": None,
        "terminate_s": None,
    }

    t0 = time.time()
    try:
        print(f"[+0.0s] launching with spike_id={SPIKE_ID}")
        resp = ec2.run_instances(
            ImageId=AMI_ID,
            InstanceType=INSTANCE_TYPE,
            MinCount=1,
            MaxCount=1,
            SubnetId=SUBNET_ID,
            SecurityGroupIds=[SG_ID],
            IamInstanceProfile={"Name": IAM_PROFILE},
            UserData=USER_DATA,
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "litellm-spike", "Value": SPIKE_ID},
                        {"Key": "Name", "Value": f"litellm-b0-spike-{SPIKE_ID[:8]}"},
                    ],
                }
            ],
        )
        instance_id = resp["Instances"][0]["InstanceId"]
        print(f"[+{time.time() - t0:.1f}s] launched {instance_id}")

        running_t0 = time.time()
        ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
        measurements["boot_to_running_s"] = time.time() - running_t0
        print(
            f"[+{time.time() - t0:.1f}s] running "
            f"(took {measurements['boot_to_running_s']:.1f}s)"
        )

        # Wait for the boot callback to fire from user-data.
        boot_elapsed = _wait_for_callback(
            callback_base, SPIKE_ID, "boot", timeout_s=180.0
        )
        if boot_elapsed is None:
            print(
                "[!] WARNING: boot callback never arrived within 180s. "
                "Check /var/log/user-data.log via SSM."
            )
        else:
            measurements["running_to_user_data_done_s"] = boot_elapsed
            print(
                f"[+{time.time() - t0:.1f}s] boot callback arrived "
                f"({boot_elapsed:.1f}s after running)"
            )

        # === WARM-ATTACH LATENCY MEASUREMENT (the critical number) ===
        # Once `running`, send a hydrate-style command via SSM RunCommand and time it.
        # SSM agent needs a moment to register with the SSM service after boot, so we
        # poll until the instance is "Online" in SSM before measuring.
        print("Waiting for SSM agent to come online...")
        ssm_online_deadline = time.time() + 180
        ssm_online = False
        while time.time() < ssm_online_deadline:
            try:
                info = ssm.describe_instance_information(
                    Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
                )
                items = info.get("InstanceInformationList", [])
                if items and items[0].get("PingStatus") == "Online":
                    ssm_online = True
                    break
            except Exception:
                pass
            time.sleep(2)

        if not ssm_online:
            # Skip warm-attach measurement explicitly so the SUMMARY line still
            # prints with a null measurement, rather than crashing inside
            # send_command with TargetNotConnected.
            print(
                "[!] SSM agent never registered as Online within 180s — "
                "skipping warm-attach measurement"
            )
            measurements["warm_attach_ssm_ms"] = None
            return 1

        print("Measuring warm-attach latency via SSM RunCommand...")
        attach_t0 = time.time()
        cmd = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={
                "commands": [
                    f"export LITELLM_SESSION_ID={SPIKE_ID}",
                    f"export LITELLM_BASE_URL='{CALLBACK_URL}'",
                    f"curl -fsS -X POST \"{CALLBACK_URL}\" -H 'Content-Type: application/json' "
                    f'-d \'{{"phase":"hydrate","spike_id":"{SPIKE_ID}"}}\'',
                ]
            },
        )
        cmd_id = cmd["Command"]["CommandId"]
        # Poll for SSM command completion (this is the metric we care about).
        inv = None
        for _ in range(120):  # 30s at 0.25s intervals
            try:
                inv = ssm.get_command_invocation(
                    CommandId=cmd_id, InstanceId=instance_id
                )
                if inv["Status"] in ("Success", "Failed", "TimedOut", "Cancelled"):
                    break
            except ssm.exceptions.InvocationDoesNotExist:
                # Command isn't fully scheduled yet; keep polling.
                pass
            time.sleep(0.25)
        attach_elapsed = time.time() - attach_t0
        measurements["warm_attach_ssm_ms"] = attach_elapsed * 1000.0
        status = inv["Status"] if inv else "NoInvocation"
        print(
            f"[+{attach_elapsed * 1000:.0f}ms] warm-attach via SSM: {status}  "
            f"<-- decides B2 hydrate transport"
        )

        # Confirm the hydrate callback arrived too (sanity check the command actually ran).
        hydrate_elapsed = _wait_for_callback(
            callback_base, SPIKE_ID, "hydrate", timeout_s=20.0
        )
        if hydrate_elapsed is None:
            print(
                "[!] WARNING: hydrate callback never arrived. SSM said success "
                "but the curl inside the command may have failed."
            )
        else:
            print(f"hydrate callback arrived {hydrate_elapsed:.2f}s after SSM call")

    finally:
        # Always terminate — even on error — to avoid runaway costs.
        if instance_id:
            term_t0 = time.time()
            try:
                ec2.terminate_instances(InstanceIds=[instance_id])
                ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
                measurements["terminate_s"] = time.time() - term_t0
                print(
                    f"[+{time.time() - t0:.1f}s] terminated "
                    f"(took {measurements['terminate_s']:.1f}s)"
                )
            except Exception as e:
                print(f"[!] FAILED TO TERMINATE {instance_id}: {e}", file=sys.stderr)
                # Last-ditch: best-effort cleanup of anything tagged litellm-spike.
                try:
                    leftovers = ec2.describe_instances(
                        Filters=[
                            {"Name": "tag:litellm-spike", "Values": [SPIKE_ID]},
                            {
                                "Name": "instance-state-name",
                                "Values": ["pending", "running", "stopping", "stopped"],
                            },
                        ]
                    )
                    ids = [
                        i["InstanceId"]
                        for r in leftovers["Reservations"]
                        for i in r["Instances"]
                    ]
                    if ids:
                        ec2.terminate_instances(InstanceIds=ids)
                        print(f"[!] emergency-terminated leftovers: {ids}")
                except Exception as inner:
                    print(f"[!] cleanup also failed: {inner}", file=sys.stderr)

    # Print machine-readable summary on the last line so a wrapper script
    # can collect the timings across multiple runs.
    summary = {
        "spike_id": SPIKE_ID,
        "instance_id": instance_id,
        "measurements": measurements,
        "total_wallclock_s": time.time() - t0,
    }
    print("SUMMARY:" + json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
