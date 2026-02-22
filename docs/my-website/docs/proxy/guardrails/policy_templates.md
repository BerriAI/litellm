# Policy Templates

Policy templates provide pre-configured guardrail policies that you can use as a starting point for your organization. Instead of manually creating policies and guardrails, you can select a template that matches your use case and deploy it with one click.

## Using Policy Templates

### In the UI

1. Navigate to **Policies → Templates** tab in the LiteLLM Admin UI
2. Browse available templates (e.g., "PII Protection", "Cost Control", "HR Compliance")
3. Click **"Use Template"** on any template
4. Review the guardrails that will be created:
   - Existing guardrails are marked with a green checkmark
   - New guardrails can be selected/deselected
5. Click **"Create X Guardrails & Use Template"**
6. Review and customize the pre-filled policy form
7. Click **"Create Policy"** to save

### Workflow

```
Select Template → Review Guardrails → Create Selected → Edit Policy → Save
```

The system automatically:
- ✅ Detects which guardrails already exist
- ✅ Creates only the missing guardrails you select
- ✅ Pre-fills the policy form with template data
- ✅ Lets you customize before saving

## Available Templates

Templates are fetched from [GitHub](https://raw.githubusercontent.com/BerriAI/litellm/main/policy_templates.json) with automatic fallback to local backup.

### Current Templates

#### 1. Advanced PII Protection (Australia)
- **Complexity:** High
- **Use Case:** Comprehensive PII detection for Australian organizations
- **Guardrails:**
  - Australian tax identifiers (TFN, ABN, Medicare)
  - Australian passports
  - International PII (SSN, passports, national IDs)
  - Contact information (email, phone, address)
  - Financial data (credit cards, IBAN)
  - API credentials (AWS, GitHub, Slack) - **BLOCKS** requests
  - Network infrastructure (IP addresses)
  - Protected class information (gender, race, religion, disability, etc.)

#### 2. Baseline PII Protection
- **Complexity:** Low
- **Use Case:** Basic protection for internal tools and testing
- **Guardrails:**
  - Australian tax identifiers
  - API credentials
  - Financial data

## Creating Your Own Policy Templates

You can contribute policy templates for the entire LiteLLM community to use.

### Template Structure

Templates are defined in JSON format with the following structure:

```json
{
  "id": "unique-template-id",
  "title": "Display Title",
  "description": "Detailed description of what this template protects",
  "icon": "ShieldCheckIcon",
  "iconColor": "text-purple-500",
  "iconBg": "bg-purple-50",
  "guardrails": [
    "guardrail-name-1",
    "guardrail-name-2"
  ],
  "complexity": "Low|Medium|High",
  "guardrailDefinitions": [
    {
      "guardrail_name": "example-guardrail",
      "litellm_params": {
        "guardrail": "litellm_content_filter",
        "mode": "pre_call",
        "patterns": [
          {
            "pattern_type": "prebuilt",
            "pattern_name": "email",
            "action": "MASK"
          }
        ],
        "pattern_redaction_format": "[{pattern_name}_REDACTED]"
      },
      "guardrail_info": {
        "description": "What this guardrail does"
      }
    }
  ],
  "templateData": {
    "policy_name": "policy-name",
    "description": "Policy description",
    "guardrails_add": ["guardrail-name-1", "guardrail-name-2"],
    "guardrails_remove": []
  }
}
```

### Field Descriptions

#### Display Fields
- **id**: Unique identifier (lowercase with hyphens)
- **title**: User-facing name shown in UI
- **description**: Detailed explanation of what the template protects
- **icon**: Icon name (must be available in UI icon map)
- **iconColor**: Tailwind CSS text color class
- **iconBg**: Tailwind CSS background color class
- **guardrails**: Array of guardrail names (for display only)
- **complexity**: Badge showing difficulty ("Low", "Medium", or "High")

#### Guardrail Definitions
- **guardrailDefinitions**: Array of complete guardrail configurations
  - Each must be a valid guardrail object that can be sent to `/guardrails` POST endpoint
  - If a guardrail already exists, it will be skipped
  - Can be empty `[]` if template uses only existing guardrails

#### Policy Configuration
- **templateData**: Object that pre-fills the policy form
  - **policy_name**: Suggested name (user can edit)
  - **description**: Policy description
  - **guardrails_add**: Array of guardrail names to include
  - **guardrails_remove**: Array to remove (usually `[]` for templates)
  - **inherit**: (Optional) Parent policy name for inheritance

### Example Template

Here's a complete example for a HIPAA compliance template:

```json
{
  "id": "hipaa-compliance",
  "title": "HIPAA Compliance Policy",
  "description": "Healthcare compliance policy that masks PHI and enforces HIPAA regulations for healthcare applications.",
  "icon": "ShieldCheckIcon",
  "iconColor": "text-red-500",
  "iconBg": "bg-red-50",
  "guardrails": [
    "phi-detector",
    "medical-record-blocker",
    "patient-id-masker"
  ],
  "complexity": "High",
  "guardrailDefinitions": [
    {
      "guardrail_name": "phi-detector",
      "litellm_params": {
        "guardrail": "litellm_content_filter",
        "mode": "pre_call",
        "patterns": [
          {
            "pattern_type": "prebuilt",
            "pattern_name": "us_ssn",
            "action": "MASK"
          },
          {
            "pattern_type": "prebuilt",
            "pattern_name": "email",
            "action": "MASK"
          },
          {
            "pattern_type": "prebuilt",
            "pattern_name": "us_phone",
            "action": "MASK"
          }
        ],
        "pattern_redaction_format": "[PHI_REDACTED]"
      },
      "guardrail_info": {
        "description": "Detects and masks Protected Health Information (PHI)"
      }
    }
  ],
  "templateData": {
    "policy_name": "hipaa-compliance-policy",
    "description": "HIPAA compliance policy for healthcare applications",
    "guardrails_add": [
      "phi-detector",
      "medical-record-blocker",
      "patient-id-masker"
    ],
    "guardrails_remove": []
  }
}
```

## Contributing Templates

To contribute a policy template for everyone to use:

### Step 1: Create Your Template JSON

1. Create a JSON file following the structure above
2. Test it locally by adding it to your local `policy_templates.json`
3. Verify all guardrails work correctly
4. Ensure descriptions are clear and helpful

### Step 2: Submit a Pull Request

1. Fork the [LiteLLM repository](https://github.com/BerriAI/litellm)
2. Add your template to `policy_templates.json` at the root
3. Add your template to `litellm/policy_templates_backup.json` (keep both in sync)
4. Create a pull request with:
   - Clear description of what the template protects
   - Use case examples
   - Any relevant compliance frameworks (HIPAA, GDPR, SOC 2, etc.)

### Guidelines

**DO:**
- ✅ Use clear, descriptive names
- ✅ Include comprehensive descriptions
- ✅ Test all guardrails thoroughly
- ✅ Document pattern sources (e.g., "Based on NIST guidelines")
- ✅ Group related guardrails logically
- ✅ Consider different complexity levels

**DON'T:**
- ❌ Include credentials or secrets
- ❌ Use overly broad patterns that may have false positives
- ❌ Duplicate existing templates
- ❌ Use custom code without thorough testing

## Using Templates Offline

For air-gapped or offline deployments, set the environment variable:

```bash
export LITELLM_LOCAL_POLICY_TEMPLATES=true
```

This forces the system to use the local backup (`litellm/policy_templates_backup.json`) instead of fetching from GitHub.

## Template Sources

- **GitHub (default):** https://raw.githubusercontent.com/BerriAI/litellm/main/policy_templates.json
- **Local backup:** `litellm/policy_templates_backup.json`

Templates are automatically fetched from GitHub on each request, with fallback to local backup on any failure.

## Available Pattern Types

When creating guardrails for templates, you can use these prebuilt patterns:

### Identity Documents
- `passport_australia`, `passport_us`, `passport_uk`, `passport_germany`, etc.
- `us_ssn`, `us_ssn_no_dash`
- `au_tfn`, `au_abn`, `au_medicare`
- `nl_bsn_contextual`
- `br_cpf`, `br_rg`, `br_cnpj`

### Financial
- `visa`, `mastercard`, `amex`, `discover`, `credit_card`
- `iban`

### Contact Information
- `email`
- `us_phone`, `br_phone_landline`, `br_phone_mobile`
- `street_address`
- `br_cep` (Brazilian postal code)

### Credentials
- `aws_access_key`, `aws_secret_key`
- `github_token`
- `slack_token`
- `generic_api_key`

### Network
- `ipv4`, `ipv6`

### Protected Class
- `gender_sexual_orientation`
- `race_ethnicity_national_origin`
- `religion`
- `age_discrimination`
- `disability`
- `marital_family_status`
- `military_status`
- `public_assistance`

See the [full patterns list](https://github.com/BerriAI/litellm/blob/main/litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/patterns.json) for all available patterns.

## Related Docs

- [Guardrail Policies](./guardrail_policies)
- [Policy Tags](./policy_tags)
- [Content Filter Patterns](../hooks/content_filter)
- [Custom Code Guardrails](../hooks/custom_code)
