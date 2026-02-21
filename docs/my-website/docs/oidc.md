# [BETA] OpenID Connect (OIDC)
LiteLLM supports using OpenID Connect (OIDC) for authentication to upstream services . This allows you to avoid storing sensitive credentials in your configuration files.

:::info

This feature is in Beta

:::


## OIDC Identity Provider (IdP)

LiteLLM supports the following OIDC identity providers:

| Provider                 | Config Name  | Custom Audiences |
| -------------------------| ------------ | ---------------- |
| Google Cloud Run         | `google`     | Yes              |
| CircleCI v1              | `circleci`   | No               |
| CircleCI v2              | `circleci_v2`| No               |
| GitHub Actions           | `github`     | Yes              |
| Azure Kubernetes Service | `azure`      | No               |
| Azure AD                 | `azure`      | Yes              |
| File                     | `file`       | No               |
| Environment Variable     | `env`        | No               |
| Environment Path         | `env_path`   | No               |

If you would like to use a different OIDC provider, please open an issue on GitHub.

:::tip

Do not use the `file`, `env`, or `env_path` providers unless you know what you're doing, and you are sure none of the other providers will work for your use-case. Hint: they probably will.

:::

## OIDC Connect Relying Party (RP)

LiteLLM supports the following OIDC relying parties / clients:

- Amazon Bedrock
- Azure OpenAI
- _(Coming soon) Google Cloud Vertex AI_


### Configuring OIDC

Wherever a secret key can be used, OIDC can be used in-place. The general format is:

```
oidc/config_name_here/audience_here
```

For providers that do not use the `audience` parameter, you can (and should) omit it:

```
oidc/config_name_here/
```

#### Unofficial Providers (not recommended)

For the unofficial `file` provider, you can use the following format:

```
oidc/file/home/user/dave/this_is_a_file_with_a_token.txt
```

For the unofficial `env`, use the following format, where `SECRET_TOKEN` is the name of the environment variable that contains the token:

```
oidc/env/SECRET_TOKEN
```

For the unofficial `env_path`, use the following format, where `SECRET_TOKEN` is the name of the environment variable that contains the path to the file with the token:

```
oidc/env_path/SECRET_TOKEN
```

:::tip

If you are tempted to use oidc/env_path/AZURE_FEDERATED_TOKEN_FILE, don't do that. Instead, use `oidc/azure/`, as this will ensure continued support from LiteLLM if Azure changes their OIDC configuration and/or adds new features.

:::

## Examples

### Google Cloud Run -> Amazon Bedrock

```yaml
model_list:
  - model_name: claude-3-haiku-20240307
    litellm_params:
      model: bedrock/anthropic.claude-3-haiku-20240307-v1:0
      aws_region_name: us-west-2
      aws_session_name: "litellm"
      aws_role_name: "arn:aws:iam::YOUR_THING_HERE:role/litellm-google-demo"
      aws_web_identity_token: "oidc/google/https://example.com"
```

### CircleCI v2 -> Amazon Bedrock

```yaml
model_list:
  - model_name: command-r
    litellm_params:
      model: bedrock/cohere.command-r-v1:0
      aws_region_name: us-west-2
      aws_session_name: "my-test-session"
      aws_role_name: "arn:aws:iam::335785316107:role/litellm-github-unit-tests-circleci"
      aws_web_identity_token: "oidc/example-provider/"
```

#### Amazon IAM Role Configuration for CircleCI v2 -> Bedrock

The configuration below is only an example. You should adjust the permissions and trust relationship to match your specific use case.

Permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                "arn:aws:bedrock:*::foundation-model/cohere.command-r-v1:0"
            ]
        }
    ]
}
```

See https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html for more examples. 

Trust Relationship:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::335785316107:oidc-provider/oidc.circleci.com/org/c5a99188-154f-4f69-8da2-b442b1bf78dd"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "oidc.circleci.com/org/c5a99188-154f-4f69-8da2-b442b1bf78dd:aud": "c5a99188-154f-4f69-8da2-b442b1bf78dd"
                },
                "ForAnyValue:StringLike": {
                    "oidc.circleci.com/org/c5a99188-154f-4f69-8da2-b442b1bf78dd:sub": [
                        "org/c5a99188-154f-4f69-8da2-b442b1bf78dd/project/*/user/*/vcs-origin/github.com/BerriAI/litellm/vcs-ref/refs/heads/main",
                        "org/c5a99188-154f-4f69-8da2-b442b1bf78dd/project/*/user/*/vcs-origin/github.com/BerriAI/litellm/vcs-ref/refs/heads/litellm_*"
                    ]
                }
            }
        }
    ]
}
```

This trust relationship restricts CircleCI to only assume the role on the main branch and branches that start with `litellm_`.

For CircleCI (v1 and v2), you also need to add your organization's OIDC provider in your AWS IAM settings. See https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-idp_oidc.html for more information.

:::tip

You should _never_ need to create an IAM user. If you did, you're not using OIDC correctly. You should only be creating a role with permissions and a trust relationship to your OIDC provider.

:::


### Google Cloud Run -> Azure OpenAI

```yaml
model_list:
  - model_name: gpt-4o-2024-05-13
    litellm_params:
      model: azure/gpt-4o-2024-05-13
      azure_ad_token: "oidc/google/https://example.com"
      api_version: "2024-06-01"
      api_base: "https://demo-here.openai.azure.com"
    model_info:
      base_model: azure/gpt-4o-2024-05-13
```

For Azure OpenAI, you need to define `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, and optionally `AZURE_AUTHORITY_HOST` in your environment.

```bash
export AZURE_CLIENT_ID="91a43c21-cf21-4f34-9085-331015ea4f91" # Azure AD Application (Client) ID
export AZURE_TENANT_ID="f3b1cf79-eba8-40c3-8120-cb26aca169c2" # Will be the same across of all your Azure AD applications
export AZURE_AUTHORITY_HOST="https://login.microsoftonline.com" # 👈 Optional, defaults to "https://login.microsoftonline.com"
```

:::tip

You can find `AZURE_CLIENT_ID` by visiting `https://login.microsoftonline.com/YOUR_DOMAIN_HERE/v2.0/.well-known/openid-configuration` and looking for the UUID in the `issuer` field.

:::


:::tip

Don't set `AZURE_AUTHORITY_HOST` in your environment unless you need to override the default value. This way, if the default value changes in the future, you won't need to update your environment.

:::


:::tip

By default, Azure AD applications use the audience `api://AzureADTokenExchange`. We recommend setting the audience to something more specific to your application.

:::


#### Azure AD Application Configuration

Unfortunately, Azure is bit more complicated to set up than other OIDC relying parties like AWS. Basically, you have to:

1. Create an Azure application.
2. Add a federated credential for the OIDC IdP you're using (e.g. Google Cloud Run).
3. Add the Azure application to resource group that contains the Azure OpenAI resource(s).
4. Give the Azure application the necessary role to access the Azure OpenAI resource(s).

The custom role below is the recommended minimum permissions for the Azure application to access Azure OpenAI resources. You should adjust the permissions to match your specific use case.

```json
{
    "id": "/subscriptions/24ebb700-ec2f-417f-afad-78fe15dcc91f/providers/Microsoft.Authorization/roleDefinitions/baf42808-99ff-466d-b9da-f95bb0422c5f",
    "properties": {
        "roleName": "invoke-only",
        "description": "",
        "assignableScopes": [
            "/subscriptions/24ebb700-ec2f-417f-afad-78fe15dcc91f/resourceGroups/your-openai-group-name"
        ],
        "permissions": [
            {
                "actions": [],
                "notActions": [],
                "dataActions": [
                    "Microsoft.CognitiveServices/accounts/OpenAI/deployments/audio/action",
                    "Microsoft.CognitiveServices/accounts/OpenAI/deployments/search/action",
                    "Microsoft.CognitiveServices/accounts/OpenAI/deployments/completions/action",
                    "Microsoft.CognitiveServices/accounts/OpenAI/deployments/chat/completions/action",
                    "Microsoft.CognitiveServices/accounts/OpenAI/deployments/extensions/chat/completions/action",
                    "Microsoft.CognitiveServices/accounts/OpenAI/deployments/embeddings/action",
                    "Microsoft.CognitiveServices/accounts/OpenAI/images/generations/action"
                ],
                "notDataActions": []
            }
        ]
    }
}
```

_Note: Your UUIDs will be different._

Please contact us for paid enterprise support if you need help setting up Azure AD applications.

### Azure AD -> Amazon Bedrock
```yaml
model list:
  - model_name: aws/claude-3-5-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
      aws_region_name: "eu-central-1"
      aws_role_name: "arn:aws:iam::12345678:role/bedrock-role"
      aws_web_identity_token: "oidc/azure/api://123-456-789-9d04"
      aws_session_name: "litellm-session"
```
