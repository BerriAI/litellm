# Zscaler AI Guard

## Overview
Zscaler AI guard enforces security policies for all traffic towards AI sites, models and applications. The AI guard is part of the Zero Trust Exchange and provides a comprehensive platform for visibility, control and deep packet inspection of AI prompts.

## 1. Setup guardrails policy on Zscaler AI Guard
Setup guardrails policy on Zscaler AI Guard, and get your ZSCALER_AI_GUARD_API_KEY, ZSCALER_AI_GUARD_POLICY_ID

## 2. Define Zscaler AI Guard in `config.yaml`

You can define Zscaler AI Guard settings directly in your LiteLLM `config.yaml` file.

### Example Configuration: 
Set ZSCALER_AI_GUARD_API_KEY, ZSCALER_AI_GUARD_POLICY_ID, ZSCALER_AI_GUARD_URL as enviroment variables

```yaml
guardrails:
  - guardrail_name: "zscaler-ai-guard-during-guard"
    litellm_params:
      guardrail: zscaler_ai_guard
      mode: "during_call"                  
      api_key: os.environ/ZSCALER_AI_GUARD_API_KEY
      api_base: os.environ/ZSCALER_AI_GUARD_URL  
      policy_id: os.environ/ZSCALER_AI_GUARD_POLICY_ID
  - guardrail_name: "zscaler-ai-guard-post-guard"
    litellm_params:
      guardrail: zscaler_ai_guard
      mode: "post_call"                   
      api_key: os.environ/ZSCALER_AI_GUARD_API_KEY
      api_base: os.environ/ZSCALER_AI_GUARD_URL  
      policy_id: os.environ/ZSCALER_AI_GUARD_POLICY_ID
```

## 3. Test request 

Expect this to fail since since `ishaan@berri.ai` in the request is PII

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-npnwjPQciVRok5yNZgKmFQ" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi my email is ishaan@berri.ai"}
    ],
    "guardrails": ["zscaler-ai-guard-during-guard", "zscaler-ai-guard-post-guard"]
  }'
```

## 4. Behavior on Violations

### Prompt is Blocked
When input violates Zscaler AI Guard policies, it returns:
- **HTTP Status**: 400
- **Error Type**: `Guardrail Policy Violation`
- **blocking_info**: 
   - `transactionId`: Zscaler AI Guard transactionId for debugging
   - `message`: Prompt or LLM response is blocked
   - `blockingDetectors`: the list of Zscaler AI Guard detectors that block the request

#### Example Response
```json
{
   "error": {
      "error_type": "Guardrail Policy Violation",
      "blocking_info": {
         "transactionId": "1234abcd-5678-efgh-9101-ijklmnopqr",
         "message": "Prompt violates Zscaler AI Guard policy.",
         "blockingDetectors": ["toxicity"]
      }
   },
   "type": "None",
   "param": "None",
   "code": "400"
}
```

### LLM response Blocked
When output violates Zscaler AI Guard policies, it returns:
- **HTTP Status**: 400
- **Error Type**: `Guardrail Policy Violation`
- **blocking_info**: 
   - `transactionId`: Zscaler AI Guard transactionId for debuging
   - `message`: Prompt or LLM response is blocked
   - `blockingDetectors`: the list of Zscaler AI Guard detectors that block the request

#### Example Response
```json
{
   "error": {
      "error_type": "Guardrail Policy Violation",
      "blocking_info": {
         "transactionId": "5678abcd-9101-efgh-1234-ijklmnopqr",
         "message": "LLM response violates Zscaler AI Guard policy.",
         "blockingDetectors": ["toxicity"]
      }
   },
   "type": "None",
   "param": "None",
   "code": "400"
}
```


## 5. Error Handling for Service Issues

In cases where Zscaler AI Guard encounters operational issues, it returns:
- **HTTP Status**: 500
- **Error Type**: `Guardrail Service Operational Issue`
- **reason**: the detailed reason 

#### Example Response
```json
{
   "error": {
      "error_type": "Zscaler AI Guard Service Operational Issue",
      "reason": "Action field in response is None, expecting 'ALLOW', 'BLOCK' or 'DETECT."
   },
   "type": "None",
   "param": "None",
   "code": "500"
}
```

