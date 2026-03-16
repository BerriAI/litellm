import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Bedrock Guardrails

:::tip ⚡️
If you haven't set up or authenticated your Bedrock provider yet, see the [Bedrock Provider Setup & Authentication Guide](../../providers/bedrock.md).
:::

:::info LiteLLM on GCP?
If your LiteLLM instance runs on GCP (Cloud Run, GKE, or Compute Engine) and you want to use Bedrock guardrails without storing AWS keys, see [Bedrock Guardrails with OIDC (GCP Deployment)](#bedrock-guardrails-with-oidc-gcp-deployment).
:::

LiteLLM supports Bedrock guardrails via the [Bedrock ApplyGuardrail API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html). 

## Quick Start
### 1. Define Guardrails on your LiteLLM config.yaml 

Define your guardrails under the `guardrails` section
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "bedrock-pre-guard"
    litellm_params:
      guardrail: bedrock  # supported values: "aporia", "bedrock", "lakera"
      mode: "during_call"
      guardrailIdentifier: ff6ujrregl1q      # your guardrail ID on bedrock
      guardrailVersion: "DRAFT"              # your guardrail version on bedrock
      aws_region_name: os.environ/AWS_REGION # region guardrail is defined
      aws_role_name: os.environ/AWS_ROLE_ARN # your role with permissions to use the guardrail
  
```

#### Supported values for `mode`

- `pre_call` Run **before** LLM call, on **input**
- `post_call` Run **after** LLM call, on **input & output**
- `during_call` Run **during** LLM call, on **input** Same as `pre_call` but runs in parallel as LLM call.  Response not returned until guardrail check completes

### 2. Start LiteLLM Gateway 


```shell
litellm --config config.yaml --detailed_debug
```

### 3. Test request 

**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

<Tabs>
<TabItem label="Unsuccessful call" value = "not-allowed">

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
    "guardrails": ["bedrock-pre-guard"]
  }'
```

Expected response on failure

```shell
{
  "error": {
    "message": {
      "error": "Violated guardrail policy",
      "bedrock_guardrail_response": {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
          {
            "topicPolicy": {
              "topics": [
                {
                  "action": "BLOCKED",
                  "name": "Coffee",
                  "type": "DENY"
                }
              ]
            }
          }
        ],
        "blockedResponse": "Sorry, the model cannot answer this question. coffee guardrail applied ",
        "output": [
          {
            "text": "Sorry, the model cannot answer this question. coffee guardrail applied "
          }
        ],
        "outputs": [
          {
            "text": "Sorry, the model cannot answer this question. coffee guardrail applied "
          }
        ],
        "usage": {
          "contentPolicyUnits": 0,
          "contextualGroundingPolicyUnits": 0,
          "sensitiveInformationPolicyFreeUnits": 0,
          "sensitiveInformationPolicyUnits": 0,
          "topicPolicyUnits": 1,
          "wordPolicyUnits": 0
        }
      }
    },
    "type": "None",
    "param": "None",
    "code": "400"
  }
}

```

</TabItem>

<TabItem label="Successful Call " value = "allowed">

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-npnwjPQciVRok5yNZgKmFQ" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi what is the weather"}
    ],
    "guardrails": ["bedrock-pre-guard"]
  }'
```

</TabItem>


</Tabs>

## Bedrock Guardrails with OIDC (GCP Deployment)

When your LiteLLM instance runs on GCP (Cloud Run, GKE, or Compute Engine) and your Bedrock guardrails are in AWS, you can use **OIDC federation** to authenticate without storing AWS keys. This guide walks you through setup via the LiteLLM UI.

:::info Prerequisites
- LiteLLM instance deployed on GCP (Cloud Run, GKE, or Compute Engine)
- Bedrock guardrail created in your AWS account
- Access to the LiteLLM UI and your AWS account
:::

### Part 1: AWS Setup

#### Step 1: Create an IAM role for OIDC

1. In **AWS Console** → **IAM** → **Roles** → **Create role**
2. Select **Web identity** as the trusted entity type
3. For **Identity provider**, select **Google**
4. For **Audience**, enter your LiteLLM instance URL (e.g. `https://litellm-proxy-xyz123-uc.a.run.app`)
5. Click **Next**
6. Attach the permissions policy (see Step 2)
7. Name the role (e.g. `litellm-bedrock-guardrails`) and create it
8. Copy the **Role ARN** (e.g. `arn:aws:iam::123456789012:role/litellm-bedrock-guardrails`)

#### Step 2: Add permissions for Bedrock guardrails

1. Open the role you created → **Add permissions** → **Create inline policy**
2. Switch to the **JSON** tab and paste:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockGuardrails",
      "Effect": "Allow",
      "Action": [
        "bedrock:ApplyGuardrail"
      ],
      "Resource": [
        "arn:aws:bedrock:us-east-1:YOUR_ACCOUNT_ID:guardrail/YOUR_GUARDRAIL_ID"
      ]
    }
  ]
}
```

3. Replace:
   - `YOUR_ACCOUNT_ID` with your AWS account ID
   - `us-east-1` with your Bedrock region if different
   - `YOUR_GUARDRAIL_ID` with your Bedrock guardrail ID (or use `arn:aws:bedrock:*:YOUR_ACCOUNT_ID:guardrail/*` for all guardrails)
4. Name the policy (e.g. `BedrockGuardrailAccess`) and create it

#### Step 3: Get your Bedrock guardrail details

1. In **AWS Console** → **Amazon Bedrock** → **Guardrails**
2. Open your guardrail
3. Note the **Guardrail ID** (e.g. `ff6ujrregl1q`), **Version** (e.g. `DRAFT`), and **Region** (e.g. `us-east-1`)

### Part 2: Add the Guardrail in the LiteLLM UI

#### Step 1: Open the Guardrails section

1. Log in to your LiteLLM instance (e.g. `https://your-litellm-instance.run.app`)
2. Go to **Guardrails** in the sidebar
3. Click **Add Guardrail**

#### Step 2: Select Bedrock

1. Choose **AWS Bedrock Guardrails**
2. Click **Next**

#### Step 3: Fill in the guardrail configuration

| Field | Value |
|-------|-------|
| **Guardrail Name** | e.g. `bedrock-content-guard` |
| **Mode** | `pre_call`, `during_call`, or `post_call` |
| **Guardrail Identifier** | Your Bedrock guardrail ID (e.g. `ff6ujrregl1q`) |
| **Guardrail Version** | e.g. `DRAFT` or your version number |
| **AWS Region Name** | Region where your guardrail lives (e.g. `us-east-1`) |
| **AWS Role Name** | Full IAM role ARN (e.g. `arn:aws:iam::123456789012:role/litellm-bedrock-guardrails`) |
| **AWS Session Name** | e.g. `litellm` |
| **AWS Web Identity Token** | `oidc/google/` + your LiteLLM instance URL (e.g. `oidc/google/https://your-litellm-instance.run.app`) |

:::tip Instance URL as audience
Use the same URL you use to access the LiteLLM UI. It must match exactly what you entered in the AWS trust policy.
:::

#### Step 4: Save and enable

1. Click **Create** or **Save**
2. Enable the guardrail on the models you want to protect

### Example values

| Field | Example |
|-------|---------|
| LiteLLM instance URL | `https://litellm-proxy-abc123-uc.a.run.app` |
| AWS Web Identity Token | `oidc/google/https://litellm-proxy-abc123-uc.a.run.app` |
| AWS Role Name | `arn:aws:iam::123456789012:role/litellm-bedrock-guardrails` |
| Guardrail Identifier | `ff6ujrregl1q` |
| Guardrail Version | `DRAFT` |
| AWS Region Name | `us-east-1` |

### Troubleshooting

| Issue | Action |
|-------|--------|
| "OIDC token could not be retrieved" | Ensure LiteLLM runs on GCP (Cloud Run, GKE, or Compute Engine) so the metadata server is available |
| "Access Denied" from AWS | Verify the IAM role trust policy uses `accounts.google.com` and the audience matches your instance URL exactly |
| "Guardrail not found" | Double-check guardrail ID, version, and region in AWS Bedrock |
| `AssumeRoleWithWebIdentity` errors | Ensure the trust policy audience matches the value in **AWS Web Identity Token** (the part after `oidc/google/`) |

### Config.yaml equivalent

The same setup can be defined in `config.yaml`:

```yaml
guardrails:
  - guardrail_name: "bedrock-pre-guard"
    litellm_params:
      guardrail: bedrock
      mode: "pre_call"
      guardrailIdentifier: ff6ujrregl1q
      guardrailVersion: "DRAFT"
      aws_region_name: "us-east-1"
      aws_role_name: "arn:aws:iam::123456789012:role/litellm-bedrock-guardrails"
      aws_session_name: "litellm"
      aws_web_identity_token: "oidc/google/https://your-litellm-instance.run.app"
```

For more on OIDC with LiteLLM, see the [OIDC documentation](../../oidc).

## PII Masking with Bedrock Guardrails

Bedrock guardrails support PII detection and masking capabilities. To enable this feature, you need to:

1. Set `mode` to `pre_call` to run the guardrail check before the LLM call
2. Enable masking by setting `mask_request_content` and/or `mask_response_content` to `true`

Here's how to configure it in your config.yaml:

```yaml showLineNumbers title="litellm proxy config.yaml"
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY
  
guardrails:
  - guardrail_name: "bedrock-pre-guard"
    litellm_params:
      guardrail: bedrock
      mode: "pre_call"  # Important: must use pre_call mode for masking
      guardrailIdentifier: wf0hkdb5x07f
      guardrailVersion: "DRAFT"
      aws_region_name: os.environ/AWS_REGION
      aws_role_name: os.environ/AWS_ROLE_ARN
      mask_request_content: true    # Enable masking in user requests
      mask_response_content: true   # Enable masking in model responses
```

With this configuration, when the bedrock guardrail intervenes, litellm will read the masked output from the guardrail and send it to the model.

### Example Usage

When enabled, PII will be automatically masked in the text. For example, if a user sends:

```
My email is john.doe@example.com and my phone number is 555-123-4567
```

The text sent to the model might be masked as:

```
My email is [EMAIL] and my phone number is [PHONE_NUMBER]
```

This helps protect sensitive information while still allowing the model to understand the context of the request.

## Experimental: Only Send Latest User Message

When you're chaining long conversations through Bedrock guardrails, you can opt into a lighter, experimental behavior by setting `experimental_use_latest_role_message_only: true` in the guardrail's `litellm_params`. When enabled, LiteLLM only sends the most recent `user` message (or assistant output during post-call checks) to Bedrock, which:

- prevents unintended blocks on older system/dev messages
- keeps Bedrock payloads smaller, reducing latency and cost
- applies to proxy hooks (`pre_call`, `during_call`) and the `/guardrails/apply_guardrail` testing endpoint

```yaml showLineNumbers title="litellm proxy config.yaml"
guardrails:
  - guardrail_name: "bedrock-pre-guard"
    litellm_params:
      guardrail: bedrock
      mode: "pre_call"
      guardrailIdentifier: wf0hkdb5x07f
      guardrailVersion: "DRAFT"
      aws_region_name: os.environ/AWS_REGION
      experimental_use_latest_role_message_only: true  # NEW
```

> ⚠️ This flag is currently experimental and defaults to `false` to preserve the legacy behavior (entire message history). We'll be listening to user feedback to decide if this becomes the default or rolls out more broadly.

## Disabling Exceptions on Bedrock BLOCK

By default, when Bedrock guardrails block content, LiteLLM raises an HTTP 400 exception. However, you can disable this behavior by setting `disable_exception_on_block: true`. This is particularly useful when integrating with **OpenWebUI**, where exceptions can interrupt the chat flow and break the user experience.

When exceptions are disabled, instead of receiving an error, you'll get a successful response containing the Bedrock guardrail's modified/blocked output.

### Configuration

Add `disable_exception_on_block: true` to your guardrail configuration:

```yaml showLineNumbers title="litellm proxy config.yaml"
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "bedrock-guardrail"
    litellm_params:
      guardrail: bedrock
      mode: "post_call"
      guardrailIdentifier: ff6ujrregl1q
      guardrailVersion: "DRAFT"
      aws_region_name: os.environ/AWS_REGION
      aws_role_name: os.environ/AWS_ROLE_ARN
      disable_exception_on_block: true  # Prevents exceptions when content is blocked
```

### Behavior Comparison

<Tabs>
<TabItem label="With Exceptions (Default)" value="with-exceptions">

When `disable_exception_on_block: false` (default):

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-npnwjPQciVRok5yNZgKmFQ" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "How do I make explosives?"}
    ],
    "guardrails": ["bedrock-guardrail"]
  }'
```

**Response: HTTP 400 Error**
```json
{
  "error": {
    "message": {
      "error": "Violated guardrail policy",
      "bedrock_guardrail_response": {
        "action": "GUARDRAIL_INTERVENED",
        "blockedResponse": "I can't provide information on creating explosives.",
        // ... additional details
      }
    },
    "type": "None",
    "param": "None", 
    "code": "400"
  }
}
```

</TabItem>

<TabItem label="Without Exceptions" value="without-exceptions">

When `disable_exception_on_block: true`:

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-npnwjPQciVRok5yNZgKmFQ" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "How do I make explosives?"}
    ],
    "guardrails": ["bedrock-guardrail"]
  }'
```

**Response: HTTP 200 Success**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "I can't provide information on creating explosives."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 12,
    "total_tokens": 22
  }
}
```

</TabItem>
</Tabs>
