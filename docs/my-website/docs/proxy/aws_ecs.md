````markdown
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Deploy LiteLLM Proxy on AWS ECS

Deploy LiteLLM Proxy as a scalable containerized service using Amazon Elastic Container Service (ECS) with AWS Fargate for serverless container orchestration.

## Overview

Amazon ECS provides fully managed container orchestration that allows you to deploy, manage, and scale containerized applications. This guide covers deploying LiteLLM Proxy using:

- **AWS Fargate**: Serverless compute for containers (recommended)
- **EC2**: Self-managed compute instances

## Prerequisites

Before you begin, ensure you have:

- AWS CLI configured with appropriate permissions
- Docker installed locally
- LiteLLM configuration file ready
- Required IAM roles and permissions (see setup below)

## Required IAM Setup

### Step 1: Create a Task Execution Role

Create this role in the [IAM Console](https://console.aws.amazon.com/iam/) → **Roles** → **Create role**:

**Role Configuration:**
- **Trusted entity**: AWS service → Elastic Container Service → Elastic Container Service Task

**Required Policies to Attach:**
- ✅ `AmazonECSTaskExecutionRolePolicy` (required for all ECS tasks)
- ✅ `SecretsManagerReadWrite` (if using AWS Secrets Manager for API keys)
- ✅ `AmazonSSMReadOnlyAccess` (if using Systems Manager parameters)

### Step 2: Create Task Role (Recommended)

Create this role for your LiteLLM application permissions:

**Role Configuration:**
- **Trusted entity**: AWS service → Elastic Container Service → Elastic Container Service Task
- **Role name**: `litellm-task-role`

**Custom Policy to Create and Attach:**

Navigate to Policies -> Create Policy
Create a custom policy called `LiteLLMTaskPolicy` with these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::<s3-litellm-bucket>",
        "arn:aws:s3:::<s3-litellm-bucket>/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData",
        "xray:PutTraceSegments",
        "xray:PutTelemetryRecords"
      ],
      "Resource": "*"
    }
  ]
}
```

**Policy Usage:**
- **S3 permissions**: If storing config files in S3
- **Bedrock permissions**: If using AWS Bedrock models
- **CloudWatch/X-Ray**: For monitoring and distributed tracing

### Step 4: Your User Permissions

Ensure your AWS user/role has these managed policies attached in IAM:

**Required AWS Managed Policies:**
- ✅ `AmazonECS_FullAccess` (for ECS operations)
- ✅ `IAMReadOnlyAccess` (to pass roles to ECS)
- ✅ `AmazonEC2ReadOnlyAccess` (for VPC/subnet information)
- ✅ `ElasticLoadBalancingFullAccess` (if using load balancers)
- ✅ `ApplicationAutoScalingFullAccess` (for auto scaling)
- ✅ `CloudWatchFullAccess` (for logging and monitoring)

**Or create a custom policy with these permissions:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecs:*",
        "iam:PassRole",
        "iam:GetRole",
        "iam:ListAttachedRolePolicies",
        "ec2:DescribeVpcs",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "elasticloadbalancing:*",
        "application-autoscaling:*",
        "cloudwatch:*",
        "logs:*"
      ],
      "Resource": "*"
    }
  ]
}
```

### Step 5: Set Up Secrets (Optional)

If storing API keys securely, create these secrets in [AWS Secrets Manager](https://console.aws.amazon.com/secretsmanager/):

**Secrets to Create:**
- **Name**: `litellm/master-key` | **Value**: Your LiteLLM master key (e.g., `sk-1234`)
- **Name**: `litellm/openai-key` | **Value**: Your OpenAI API key
- **Name**: `litellm/database-url` | **Value**: Your PostgreSQL connection string
- **Name**: `litellm/anthropic-key` | **Value**: Your Anthropic API key (if using Claude)

**Quick CLI commands if preferred:**
```bash
aws secretsmanager create-secret --name "litellm/master-key" --secret-string "sk-your-secure-master-key"
aws secretsmanager create-secret --name "litellm/openai-key" --secret-string "sk-your-openai-key"
aws secretsmanager create-secret --name "litellm/database-url" --secret-string "postgresql://user:pass@host:5432/db"
```

### Step 6: Verify IAM Setup

Before proceeding, verify your IAM configuration is correct:

```bash
# 1. Check if ECS service-linked role exists
aws iam get-role --role-name AWSServiceRoleForECS

# 2. Verify task execution role
aws iam get-role --role-name ecsTaskExecutionRole

# 3. List attached policies for execution role
aws iam list-attached-role-policies --role-name ecsTaskExecutionRole

# 4. Verify task role (if created)
aws iam get-role --role-name litellm-task-role

```

:::tip
If you encounter the error "Unable to assume the service linked role", run:
```bash
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com
```
:::

## Quick Start with Fargate

### 1. Create Task Definition

Navigate to Amazon ECS -> Task Definitions -> Create new task definition with JSON

```json
{
  "family": "litellm-proxy",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::YOUR-ACCOUNT-ID:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::YOUR-ACCOUNT-ID:role/litellm-task-role",
  "containerDefinitions": [
    {
      "name": "litellm-proxy",
      "image": "ghcr.io/berriai/litellm:main-stable",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 4000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LITELLM_MASTER_KEY",
          "value": "sk-1234"
        },
        {
            "name": "LITELLM_SALT_KEY",
            "value": "sk-123543"
        },
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:314146340344:secret:litellm-secrets-Smz6Oq:DATABASE_URL::"
        },
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:314146340344:secret:litellm-secrets-Smz6Oq:OPENAI_API_KEY::"
        }
      ],
      "logConfiguration": { // optional: if you want to monitor using cloudwatch
        "logDriver": "awslogs", 
        "options": {
          "awslogs-group": "/ecs/litellm-proxy",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
    },
      "healthCheck": {
        "command": [
          "CMD-SHELL", "wget --no-verbose --tries=1 http://localhost:4000/health/liveliness || exit 1" 
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

## Configuration Management

### Using Config File with EFS

Mount your configuration file using Amazon EFS:

1. **Create EFS File System**:
```bash
aws efs create-file-system \
    --performance-mode generalPurpose \
    --throughput-mode provisioned \
    --provisioned-throughput-in-mibps 100 \
    --tags Key=Name,Value=litellm-config
```

2. **Upload Config File to EFS**:
```bash
# Mount EFS locally and copy config
sudo mount -t efs fs-12345678:/ /mnt/efs
sudo cp litellm-config.yaml /mnt/efs/
```

3. **Update Task Definition**:
```json
{
  "volumes": [
    {
      "name": "litellm-config",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-12345678",
        "rootDirectory": "/",
        "transitEncryption": "ENABLED"
      }
    }
  ],
  "containerDefinitions": [
    {
      "mountPoints": [
        {
          "sourceVolume": "litellm-config",
          "containerPath": "/app/config",
          "readOnly": true
        }
      ],
      "command": [
        "litellm",
        "--config",
        "/app/config/litellm-config.yaml",
        "--port",
        "4000"
      ]
    }
  ]
}
```

### Using S3 Config

Store your configuration in S3 and use environment variables:

```json
{
  "environment": [
    {
      "name": "LITELLM_CONFIG_BUCKET_NAME",
      "value": "my-litellm-config-bucket"
    },
    {
      "name": "LITELLM_CONFIG_BUCKET_OBJECT_KEY",
      "value": "app/litellm-config.yaml"
    }
  ]
}
```

### 2. Register Task Definition

```bash
aws ecs register-task-definition file://litellm-task-definition.json
```

### 3. Create ECS Cluster

```bash
aws ecs create-cluster \
    --cluster-name litellm-cluster \
    --capacity-providers FARGATE \
    --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1
```

### 4. Create Service

Create a new service in UI by referencing the task name

Your deployment should start running now!


## Load Balancing & Auto Scaling

### Application Load Balancer Setup

1. **Create Target Group**:
```bash
aws elbv2 create-target-group \
    --name litellm-tg \
    --protocol HTTP \
    --port 4000 \
    --vpc-id vpc-12345678 \
    --target-type ip \
    --health-check-path /health
```

2. **Create Load Balancer**:
```bash
aws elbv2 create-load-balancer \
    --name litellm-alb \
    --subnets subnet-12345678 subnet-87654321 \
    --security-groups sg-12345678
```

3. **Create Listener**:
```bash
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:region:account:loadbalancer/app/litellm-alb/1234567890123456 \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:region:account:targetgroup/litellm-tg/1234567890123456
```

## Monitoring and Logging

### CloudWatch Setup

1. **Create Log Group**:
```bash
aws logs create-log-group --log-group-name /ecs/litellm-proxy
```

2. **Set Retention Policy**:
```bash
aws logs put-retention-policy \
    --log-group-name /ecs/litellm-proxy \
    --retention-in-days 30
```

### Custom Metrics

Add to your task definition for enhanced monitoring:

```json
{
  "environment": [
    {
      "name": "OTEL_EXPORTER_OTLP_ENDPOINT",
      "value": "https://cloudwatch-ingest.us-west-2.amazonaws.com"
    },
    {
      "name": "AWS_REGION",
      "value": "us-west-2"
    }
  ]
}
```

## Testing Your Deployment

```bash
curl -X POST http://your-public-or-private-ip/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-master-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello from ECS!"}]
  }'
```

## Troubleshooting

### Common Issues

1. **Task Startup Failures**:
   - Check CloudWatch logs for container errors
   - Verify IAM permissions for Secrets Manager
   - Ensure database connectivity

2. **Health Check Failures**:
   - Verify health check endpoint accessibility. use health/liveliness
   - Check container resource allocation
