import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 🔗 Connect AWS → GCP with Workload Identity Federation (WIF) for LiteLLM

**Last updated:** WIP — based on [PR #10210](https://github.com/BerriAI/litellm/pull/10210)  
**Applies to:** LiteLLM ≥ v1.77.x  
**Goal:** Run LiteLLM on AWS (EKS / EC2 / Lambda) and securely call GCP services (e.g., Vertex AI / Gemini) *without service-account keys.*

---

## 🧭 Overview

Workload Identity Federation (WIF) allows AWS workloads to obtain **short-lived Google Cloud tokens** using their AWS IAM role.  
LiteLLM automatically detects WIF configs (`type: external_account`) and loads them with:

~~~python
google.auth.aws.Credentials.from_info(...)
~~~

No key files, no manual rotation — credentials are exchanged securely through AWS STS → GCP STS → Service Account Impersonation.

### 🔄 Token flow

~~~text
 [AWS Role / Pod Role]
          │
          ▼
   AWS STS → signed GetCallerIdentity
          │
          ▼
 Google STS (Workload Identity Pool)
          │
          ▼
 Impersonated GCP Service Account
          │
          ▼
 Vertex AI / Gemini APIs
          │
          ▼
      LiteLLM Proxy
~~~

---

## 🧩 Prerequisites

| Requirement | Description |
|--------------|-------------|
| **AWS** | EC2, EKS, or Lambda with an IAM Role that your workload assumes |
| **GCP** | Project with APIs enabled: `sts.googleapis.com`, `iamcredentials.googleapis.com`, and your target API (e.g., `aiplatform.googleapis.com`) |
| **Permissions** | Ability to create IAM Service Accounts, WIF Pools & Providers, and grant IAM bindings |

---

## ① Get Your AWS Information

First, identify your AWS account and role details:

```bash
# Get AWS Account ID and Role ARN
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
AWS_ROLE_ARN=$(aws sts get-caller-identity --query 'Arn' --output text)
echo "AWS Account ID: $AWS_ACCOUNT_ID"
echo "AWS Role ARN: $AWS_ROLE_ARN"

# Extract role name from ARN (e.g., "arn:aws:iam::123456789:role/MyRole" → "MyRole")
AWS_ROLE_NAME=$(echo $AWS_ROLE_ARN | cut -d'/' -f2)
echo "AWS Role Name: $AWS_ROLE_NAME"
```

---

## ② Get Your GCP Information

```bash
# Set your GCP project ID
export GCP_PROJECT="your-project-id"

# Get GCP project number (needed for audience URL)
GCP_PROJECT_NUMBER=$(gcloud projects describe $GCP_PROJECT --format="value(projectNumber)")
echo "GCP Project ID: $GCP_PROJECT"
echo "GCP Project Number: $GCP_PROJECT_NUMBER"
```

---

## ③ Create a GCP Service Account

```bash
gcloud iam service-accounts create litellm-wif-sa --project $GCP_PROJECT

# Grant only required roles:
gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --member="serviceAccount:litellm-wif-sa@$GCP_PROJECT.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

> 💡 **Common roles**

| Use Case | Role Name |
|-----------|------------|
| Vertex AI (Gemini) | `roles/aiplatform.user` |
| Cloud Storage | `roles/storage.objectViewer` |
| BigQuery | `roles/bigquery.user` |

---

## ④ Create a Workload Identity Pool & AWS Provider

```bash
# Create the identity pool
gcloud iam workload-identity-pools create aws-pool \
  --project=$GCP_PROJECT --location="global" \
  --display-name="AWS Pool"

# Create the AWS provider
gcloud iam workload-identity-pools providers create-aws aws-provider \
  --project=$GCP_PROJECT --location="global" \
  --workload-identity-pool="aws-pool" \
  --account-id="$AWS_ACCOUNT_ID"
```

> 🧱 You can restrict access by specific Role ARNs or AWS account conditions using `--attribute-condition`.

---

## ⑤ Allow the Pool to Impersonate the Service Account

```bash
gcloud iam service-accounts add-iam-policy-binding \
  "litellm-wif-sa@$GCP_PROJECT.iam.gserviceaccount.com" \
  --project=$GCP_PROJECT \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$GCP_PROJECT_NUMBER/locations/global/workloadIdentityPools/aws-pool/attribute.aws_role/arn:aws:iam::$AWS_ACCOUNT_ID:role/$AWS_ROLE_NAME"
```

This binding allows the AWS role to impersonate the GCP service account via the identity pool.

---

## ⑥ Create the `external_account` JSON for AWS

Create the directory and save the configuration file:

```bash
# Create directory and set permissions
sudo mkdir -p /etc/litellm
sudo chown $USER:$USER /etc/litellm

# Create the external account JSON file
cat > /etc/litellm/gcp_aws_external_account.json << EOF
{
  "type": "external_account",
  "audience": "//iam.googleapis.com/projects/$GCP_PROJECT_NUMBER/locations/global/workloadIdentityPools/aws-pool/providers/aws-provider",
  "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
  "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/litellm-wif-sa@$GCP_PROJECT.iam.gserviceaccount.com:generateAccessToken",
  "token_url": "https://sts.googleapis.com/v1/token",
  "credential_source": {
    "environment_id": "aws1",
    "region_url": "http://169.254.169.254/latest/meta-data/placement/availability-zone",
    "url": "http://169.254.169.254/latest/meta-data/iam/security-credentials",
    "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15",
    "imdsv2_session_token_url": "http://169.254.169.254/latest/api/token"
  }
}
EOF

# Set secure permissions
chmod 600 /etc/litellm/gcp_aws_external_account.json
```

> 🔒 Uses AWS IMDS for temporary creds — never embeds a private key.

---

## 🚀 EKS Serverless (Fargate) Specific Considerations

EKS serverless pods run on AWS Fargate, which has some differences from regular EKS or EC2:

### **Key Differences for EKS Serverless:**

1. **No EC2 Instance Metadata**: Fargate pods don't have access to EC2 instance metadata
2. **Pod Identity**: Uses EKS Pod Identity instead of EC2 instance roles
3. **IMDS Access**: Limited IMDS access compared to EC2

### **EKS Serverless Configuration:**

#### **Step 1: Enable EKS Pod Identity**
```bash
# Enable pod identity for your EKS cluster
aws eks update-cluster-config \
  --name your-cluster-name \
  --pod-identity-configuration '{
    "type": "POD_IDENTITY"
  }'
```

#### **Step 2: Create Pod Identity Association**
```bash
# Create a pod identity association for your namespace
aws eks create-pod-identity-association \
  --cluster-name your-cluster-name \
  --namespace litellm \
  --service-account litellm-sa \
  --role-arn arn:aws:iam::YOUR_ACCOUNT:role/LitellmPodRole
```

#### **Step 3: Update the external_account JSON for EKS Serverless**
```json
{
  "type": "external_account",
  "audience": "//iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/aws-pool/providers/aws-provider",
  "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
  "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/litellm-wif-sa@PROJECT_ID.iam.gserviceaccount.com:generateAccessToken",
  "token_url": "https://sts.googleapis.com/v1/token",
  "credential_source": {
    "environment_id": "aws1",
    "region_url": "http://169.254.170.2/v2/metadata/placement/region",
    "url": "http://169.254.170.2/v2/credentials",
    "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15",
    "imdsv2_session_token_url": "http://169.254.170.2/v2/token"
  }
}
```

> 🔍 **Key Changes for EKS Serverless:**
> - `region_url`: Uses Fargate metadata endpoint (`169.254.170.2`)
> - `url`: Uses Fargate credentials endpoint
> - `imdsv2_session_token_url`: Uses Fargate token endpoint

#### **Step 4: Update IAM Binding for EKS Serverless**
```bash
# Use the EKS service account role ARN instead of EC2 role
gcloud iam service-accounts add-iam-policy-binding \
  "litellm-wif-sa@$GCP_PROJECT.iam.gserviceaccount.com" \
  --project=$GCP_PROJECT \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$GCP_PROJECT_NUMBER/locations/global/workloadIdentityPools/aws-pool/attribute.aws_role/arn:aws:iam::$AWS_ACCOUNT_ID:role/LitellmPodRole"
```

#### **Step 5: Deploy with EKS Service Account**
```yaml
# litellm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm
  namespace: litellm
spec:
  template:
    spec:
      serviceAccountName: litellm-sa
      containers:
      - name: litellm
        image: litellm/litellm:latest
        env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: "/etc/litellm/gcp_aws_external_account.json"
        - name: GOOGLE_CLOUD_PROJECT
          value: "$GCP_PROJECT"
        volumeMounts:
        - name: gcp-credentials
          mountPath: /etc/litellm
          readOnly: true
      volumes:
      - name: gcp-credentials
        secret:
          secretName: gcp-credentials
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: litellm-sa
  namespace: litellm
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::YOUR_ACCOUNT:role/LitellmPodRole
```

### **EKS Serverless Troubleshooting:**

| Issue | Cause | Solution |
|-------|-------|----------|
| `Unable to retrieve credentials` | Wrong metadata endpoints | Use Fargate endpoints (`169.254.170.2`) |
| `Access denied` | Pod identity not configured | Verify pod identity association |
| `Invalid role` | Wrong role ARN in binding | Use EKS service account role, not EC2 role |
| `Token refresh failed` | IMDSv2 not supported | Remove `imdsv2_session_token_url` for Fargate |

### **Environment Detection Script:**
```bash
#!/bin/bash
# Detect if running on EKS Serverless vs EC2
if curl -s --max-time 2 http://169.254.170.2/v2/metadata/placement/region > /dev/null; then
    echo "Running on EKS Serverless (Fargate)"
    export METADATA_ENDPOINT="169.254.170.2"
elif curl -s --max-time 2 http://169.254.169.254/latest/meta-data/placement/availability-zone > /dev/null; then
    echo "Running on EC2"
    export METADATA_ENDPOINT="169.254.169.254"
else
    echo "Unknown environment"
    exit 1
fi
```

---

## ⑦ Configure LiteLLM

<Tabs>
<TabItem value="env" label="Environment Variables">

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/etc/litellm/gcp_aws_external_account.json
export GOOGLE_CLOUD_PROJECT=$GCP_PROJECT
```

</TabItem>
<TabItem value="config" label="config.yaml">

```yaml
model_list:
  - model_name: gemini-2.5-pro
    litellm_params:
      model: "gemini/gemini-2.5-pro"
      vertex_project: "$GCP_PROJECT"  # Use PROJECT_ID, not PROJECT_NUMBER
      vertex_credentials: |
        {
          "type": "external_account",
          "audience": "//iam.googleapis.com/projects/$GCP_PROJECT_NUMBER/locations/global/workloadIdentityPools/aws-pool/providers/aws-provider",
          "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
          "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/litellm-wif-sa@$GCP_PROJECT.iam.gserviceaccount.com:generateAccessToken",
          "token_url": "https://sts.googleapis.com/v1/token",
          "credential_source": {
            "environment_id": "aws1",
            "region_url": "http://169.254.169.254/latest/meta-data/placement/availability-zone",
            "url": "http://169.254.169.254/latest/meta-data/iam/security-credentials",
            "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15",
            "imdsv2_session_token_url": "http://169.254.169.254/latest/api/token"
          }
        }
```

</TabItem>
</Tabs>

---

## ⑧ Verify Credentials (Recommended Test)

Run inside the AWS environment:

```python
import json
import google.auth
from google.auth.transport.requests import Request

try:
    creds, project = google.auth.load_credentials_from_file("/etc/litellm/gcp_aws_external_account.json")
    print(f"✅ Successfully loaded credentials for project: {project}")
    
    # Test token refresh
    creds.refresh(Request())
    print(f"✅ Token refresh successful: {creds.token[:20]}...")
    print(f"✅ Token expires at: {creds.expiry}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Common issues:")
    print("- Check if AWS role has proper permissions")
    print("- Verify GCP service account binding")
    print("- Ensure IMDS is accessible from your AWS environment")
```

✅ You should see a valid token — proof that AWS → GCP exchange works.

---

## ⑨ Test LiteLLM Integration

```bash
curl -X POST https://your-litellm-domain/v1/completions \
  -H "Authorization: Bearer sk-your-litellm-key" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gemini-2.5-pro",
        "messages": [{"role": "user", "content": "Hello from AWS to GCP!"}]
      }'
```

If configured correctly, the response should come from Vertex AI without any static key files.

---

## ⚠️ Troubleshooting

| Symptom | Likely Cause | Fix |
|----------|--------------|-----|
| `PERMISSION_DENIED iam.serviceAccounts.getAccessToken` | Missing `roles/iam.workloadIdentityUser` binding | Verify Step ⑤ IAM binding |
| `Unable to acquire impersonated credentials` | Audience / provider mismatch | Confirm `audience` URL matches provider path |
| Request hangs | IMDSv2 token required but not fetched | Ensure `imdsv2_session_token_url` is present or IMDSv2 disabled |
| 404 from Vertex AI | Wrong `vertex_project` | Use PROJECT_ID (not PROJECT_NUMBER) for `vertex_project` |
| `Invalid audience` | Wrong project number in audience URL | Verify `GCP_PROJECT_NUMBER` is correct |
| `Access denied` | AWS role not in IAM binding | Check the role ARN in Step ⑤ matches your actual AWS role |

---

## ✅ Security Best Practices

- Restrict provider to specific AWS Role ARNs  
- Rotate AWS Role policies regularly  
- Never distribute this JSON beyond the AWS runtime  
- Audit token exchanges in **Cloud Audit Logs → `iam.googleapis.com`**
- Use least-privilege IAM roles
- Regularly rotate AWS access keys if using temporary credentials

---

## 📚 References

- [LiteLLM PR #10210 – AWS WIF Support](https://github.com/BerriAI/litellm/pull/10210)  
- [Google IAM – Workload Identity Federation Overview](https://cloud.google.com/iam/docs/workload-identity-federation)  
- [google-auth AWS Credentials Docs](https://cloud.google.com/docs/authentication/aws)  
- [Vertex AI Permissions Reference](https://cloud.google.com/vertex-ai/docs/general/access-control)

---

**✅ Result:** Fully keyless authentication path between AWS → GCP for LiteLLM.  
No secrets in env vars, short-lived tokens only, and compliant with enterprise security guidelines.
