# Sail max_tokens check (2026-09-26)

All three requests were blocked before reaching Sail: the session's network policy denied CONNECT to api.sailresearch.com:443 (proxy answered 403). No Sail response, usage, or finish_reason was obtained, so whether the token limit is honored is untested

## 1. max_tokens only

    curl -sS -w '\nHTTP %{http_code}\n' https://api.sailresearch.com/v1/chat/completions -H "Authorization: Bearer $SAIL_API_KEY" -H "Content-Type: application/json" -d '{"model":"zai-org/GLM-5.3","messages":[{"role":"user","content":"Reply with the word ok."}],"max_tokens":16}'

Status: HTTP 000 (curl: (56) CONNECT tunnel failed, response 403). usage: n/a. finish_reason: n/a. error: network policy denial

## 2. max_completion_tokens only

    curl -sS -w '\nHTTP %{http_code}\n' https://api.sailresearch.com/v1/chat/completions -H "Authorization: Bearer $SAIL_API_KEY" -H "Content-Type: application/json" -d '{"model":"zai-org/GLM-5.3","messages":[{"role":"user","content":"Reply with the word ok."}],"max_completion_tokens":16}'

Status: HTTP 000 (curl: (56) CONNECT tunnel failed, response 403). usage: n/a. finish_reason: n/a. error: network policy denial

## 3. both max_tokens 16 and max_completion_tokens 32

    curl -sS -w '\nHTTP %{http_code}\n' https://api.sailresearch.com/v1/chat/completions -H "Authorization: Bearer $SAIL_API_KEY" -H "Content-Type: application/json" -d '{"model":"zai-org/GLM-5.3","messages":[{"role":"user","content":"Reply with the word ok."}],"max_tokens":16,"max_completion_tokens":32}'

Status: HTTP 000 (curl: (56) CONNECT tunnel failed, response 403). usage: n/a. finish_reason: n/a. error: network policy denial
