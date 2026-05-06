"""
Cloud Agents settings endpoints (Epic G / LIT-2891).

Backs the Settings -> Cloud Agents UI with four resources:
* AgentVMConfig (provider, AWS BYOC, warm pool, network access)
* AgentSecret (per-team encrypted secrets, write-only on read)
* AgentWorker (self-hosted worker registrations)
* AgentWorkerPairingToken (single-use 15-min tokens for worker install)
"""
