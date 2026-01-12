1. docker compose up
2. Start the fake LLM provider and make sure the delay is set to 10sec
3. Make the request, and wait 3 seconds:
``` bash
# Create the JSON file first
@'
{"model":"db-openai-endpoint","messages":[{"role":"user","content":"Say hello"}],"max_tokens":2000}
'@ | Out-File -FilePath request.json -Encoding utf8

# Then use it with curl
curl.exe -X POST http://localhost:4000/v1/chat/completions -H "Content-Type: application/json" -d "@request.json"
```
4. On a different terminal run:
```bash
docker kill --signal=SIGTERM (docker ps --filter "name=litellm" --format "{{.Names}}" | Select-Object -First 1)
```
5. Check if the call gets terminated or if LiteLLM waits the 10 seconds.