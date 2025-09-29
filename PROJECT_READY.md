# Project Readiness

Policy: DEV; READINESS_EXPECT=ollama,codex-agent,all_smokes_core,all_smokes_nd

Resolved endpoints:
- mini-agent: 127.0.0.1:8788
- codex-agent base: http://127.0.0.1:8788
- ollama: http://127.0.0.1:11434

## Results
- ✅ deterministic_local
- ✅ mini_agent_e2e_low
- ✅ codex_agent_router_shim
- ❌ all_smokes_core
- ❌ all_smokes_nd
- ✅ docker_smokes
- ✅ ollama_live
- ✅ codex_agent_live

Artifacts:
- local/artifacts/mvp/mvp_report.json
