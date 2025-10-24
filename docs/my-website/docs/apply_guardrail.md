import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /guardrails/apply_guardrail

Use this endpoint to directly call a guardrail configured on your LiteLLM instance. This is useful when you have services that need to directly call a guardrail.

## Supported Guardrail Types

This endpoint supports various guardrail types including:
- **Presidio** - PII detection and masking
- **Bedrock** - AWS Bedrock guardrails for content moderation
- **Lakera** - AI safety guardrails
- **Custom guardrails** - User-defined guardrails

## Configuration

### Bedrock Guardrail Configuration

To use Bedrock guardrails with the apply_guardrail endpoint, configure your guardrail in your LiteLLM config.yaml:

```yaml
guardrails:
  - guardrail_name: "bedrock-content-guard"
    litellm_params:
      guardrail: bedrock
      mode: "pre_call"
      guardrailIdentifier: "your-guardrail-id"  # Your actual Bedrock guardrail ID
      guardrailVersion: "DRAFT"  # or your version number
      aws_region_name: "us-east-1"  # Your AWS region
      aws_role_name: "your-role-arn"  # Your AWS role with Bedrock permissions
      default_on: true
```

**Required AWS Setup:**
1. Create a Bedrock guardrail in AWS Console
2. Get the guardrail ID and version
3. Ensure your AWS credentials have Bedrock permissions
4. Configure the guardrail in your LiteLLM config 


## Usage
---

<Tabs>
<TabItem value="presidio" label="Presidio PII Guardrail" default>

In this example `mask_pii` is a Presidio guardrail configured on LiteLLM.

```bash showLineNumbers title="Example calling the endpoint"
curl -X POST 'http://localhost:4000/guardrails/apply_guardrail' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "guardrail_name": "mask_pii",
    "text": "My name is John Doe and my email is john@example.com",
    "language": "en",
    "entities": ["NAME", "EMAIL"]
}'
```

</TabItem>
<TabItem value="bedrock" label="Bedrock Guardrail">

In this example `bedrock-content-guard` is a Bedrock guardrail configured on LiteLLM.

```bash showLineNumbers title="Example calling the endpoint"
curl -X POST 'http://localhost:4000/guardrails/apply_guardrail' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "guardrail_name": "bedrock-content-guard",
    "text": "This is potentially harmful content that should be blocked",
    "language": "en"
}'
```

**Note**: For Bedrock guardrails, the `entities` parameter is not used as Bedrock handles content moderation based on its own policies.

</TabItem>
</Tabs>


## Request Format
---

The request body should follow the ApplyGuardrailRequest format.

#### Example Request Body

```json
{
    "guardrail_name": "mask_pii",
    "text": "My name is John Doe and my email is john@example.com",
    "language": "en",
    "entities": ["NAME", "EMAIL"]
}
```

#### Required Fields
- **guardrail_name** (string):  
  The identifier for the guardrail to apply (e.g., "mask_pii").
- **text** (string):  
  The input text to process through the guardrail.

#### Optional Fields
- **language** (string):  
  The language of the input text (e.g., "en" for English).
- **entities** (array of strings):  
  Specific entities to process or filter (e.g., ["NAME", "EMAIL"]).

## Response Format
---

The response will contain the processed text after applying the guardrail.

#### Example Response

<Tabs>
<TabItem value="presidio" label="Presidio Response" default>

```json
{
    "response_text": "My name is [REDACTED] and my email is [REDACTED]"
}
```

</TabItem>
<TabItem value="bedrock" label="Bedrock Response">

```json
{
    "response_text": "This is potentially harmful content that should be blocked"
}
```

**Note**: If Bedrock guardrail blocks the content, the endpoint will return an error with the blocking reason.

</TabItem>
</Tabs>

#### Response Fields
- **response_text** (string):  
  The text after applying the guardrail.

#### Error Responses

If a guardrail blocks content (e.g., Bedrock guardrail), the endpoint will return an error:

```json
{
    "detail": "Content blocked by Bedrock guardrail: Content violates policy"
}
```
