# AWS Deployment for LiteLLM - Benchmark Configuration

This directory contains 1-click deployment templates for deploying LiteLLM on AWS, configured to match the [benchmark specifications](https://docs.litellm.ai/docs/benchmarks) for optimal performance.

## Benchmark Performance Targets

**Configuration:**
- 4 instances with 4 vCPUs and 8 GB RAM each
- 4 workers per instance (16 total workers)
- PostgreSQL database
- Application Load Balancer

**Expected Performance:**
- **Median latency:** ~100 ms
- **P95 latency:** ~150 ms
- **P99 latency:** ~240 ms
- **Average latency:** ~111.7 ms
- **Throughput:** ~1,170 RPS
- **LiteLLM overhead:** ~2 ms median

## Deployment Options

### Option 1: AWS ECS (Recommended - Simpler)

AWS ECS with Fargate provides a fully managed container orchestration service without needing to manage EC2 instances.

#### Prerequisites

- AWS CLI configured with appropriate credentials
- Permissions to create VPC, ECS, RDS, ALB, IAM resources

#### Quick Deploy

```bash
# Set your parameters
STACK_NAME="litellm-benchmark"
DB_PASSWORD="YourSecureDBPassword123"
MASTER_KEY="YourSecureMasterKey1234567890"

# Deploy the stack
aws cloudformation create-stack \
  --stack-name $STACK_NAME \
  --template-body file://cloudformation-ecs.yaml \
  --parameters \
    ParameterKey=DBPassword,ParameterValue=$DB_PASSWORD \
    ParameterKey=MasterKey,ParameterValue=$MASTER_KEY \
  --capabilities CAPABILITY_IAM \
  --region us-east-1

# Wait for the stack to complete (takes ~10-15 minutes)
aws cloudformation wait stack-create-complete \
  --stack-name $STACK_NAME \
  --region us-east-1

# Get the Load Balancer URL
aws cloudformation describe-stacks \
  --stack-name $STACK_NAME \
  --region us-east-1 \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' \
  --output text
```

#### Customization

You can customize the deployment by providing additional parameters:

```bash
aws cloudformation create-stack \
  --stack-name $STACK_NAME \
  --template-body file://cloudformation-ecs.yaml \
  --parameters \
    ParameterKey=DBPassword,ParameterValue=$DB_PASSWORD \
    ParameterKey=MasterKey,ParameterValue=$MASTER_KEY \
    ParameterKey=DesiredTaskCount,ParameterValue=4 \
    ParameterKey=NumWorkersPerTask,ParameterValue=4 \
    ParameterKey=TaskCPU,ParameterValue=4096 \
    ParameterKey=TaskMemory,ParameterValue=8192 \
  --capabilities CAPABILITY_IAM \
  --region us-east-1
```

### Option 2: Terraform (More Flexible)

For teams preferring Infrastructure as Code with Terraform:

```bash
cd terraform-ecs

# Initialize Terraform
terraform init

# Review the plan
terraform plan \
  -var="db_password=YourSecureDBPassword123" \
  -var="master_key=YourSecureMasterKey1234567890"

# Deploy
terraform apply \
  -var="db_password=YourSecureDBPassword123" \
  -var="master_key=YourSecureMasterKey1234567890"

# Get outputs
terraform output load_balancer_url
terraform output api_endpoint
```

## Testing Your Deployment

### 1. Health Check

```bash
LOAD_BALANCER_URL=$(aws cloudformation describe-stacks \
  --stack-name $STACK_NAME \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' \
  --output text)

curl $LOAD_BALANCER_URL/health/readiness
```

### 2. API Test

```bash
# Get your Master Key (if you forgot it)
MASTER_KEY=$(aws secretsmanager get-secret-value \
  --secret-id $STACK_NAME-master-key \
  --query SecretString \
  --output text)

# Test the API
curl -X POST "$LOAD_BALANCER_URL/v1/chat/completions" \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fake-openai-endpoint",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### 3. Load Testing (Benchmark Replication)

To replicate the benchmark results, use Locust:

```bash
# Install Locust
pip install locust

# Create a locustfile (see examples below)
# Run load test with benchmark parameters
locust -f locustfile.py \
  --host=$LOAD_BALANCER_URL \
  --users=1000 \
  --spawn-rate=500 \
  --run-time=5m \
  --headless
```

**Example Locustfile:**

```python
from locust import HttpUser, task, between
import os

class LiteLLMUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.master_key = os.environ.get("LITELLM_MASTER_KEY")

    @task
    def chat_completion(self):
        self.client.post("/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.master_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "fake-openai-endpoint",
                "messages": [{"role": "user", "content": "test"}]
            }
        )
```

## Configuration

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| DesiredTaskCount | 4 | Number of ECS tasks (instances) |
| NumWorkersPerTask | 4 | Workers per task |
| TaskCPU | 4096 | CPU units per task (4 vCPU) |
| TaskMemory | 8192 | Memory in MB per task (8 GB) |
| DBInstanceClass | db.t3.medium | RDS instance type |

### Modifying for Different Scales

**For 2 instances (reference configuration):**
```bash
--parameters \
  ParameterKey=DesiredTaskCount,ParameterValue=2 \
  ParameterKey=NumWorkersPerTask,ParameterValue=4
```

**For higher throughput (8 instances):**
```bash
--parameters \
  ParameterKey=DesiredTaskCount,ParameterValue=8 \
  ParameterKey=NumWorkersPerTask,ParameterValue=4
```

**For more powerful instances:**
```bash
--parameters \
  ParameterKey=TaskCPU,ParameterValue=8192 \
  ParameterKey=TaskMemory,ParameterValue=16384
```

## Monitoring

### CloudWatch Logs

View logs from your ECS tasks:

```bash
aws logs tail /ecs/$STACK_NAME-litellm --follow
```

### CloudWatch Metrics

Key metrics to monitor:
- **ECS:** CPUUtilization, MemoryUtilization
- **ALB:** TargetResponseTime, RequestCount, HealthyHostCount
- **RDS:** DatabaseConnections, CPUUtilization, FreeableMemory

### LiteLLM Overhead Monitoring

LiteLLM reports its overhead in the `x-litellm-overhead-duration-ms` response header. Monitor this to track proxy performance.

## Cost Estimation

**Monthly costs (us-east-1, approximate):**

| Resource | Configuration | Monthly Cost |
|----------|---------------|--------------|
| ECS Fargate | 4 tasks × 4 vCPU × 8 GB | ~$350 |
| RDS PostgreSQL | db.t3.medium, 100 GB | ~$60 |
| Application Load Balancer | 1 ALB | ~$20 |
| Data Transfer | Varies by usage | ~$10-50 |
| **Total** | | **~$440-460/month** |

**Cost optimization tips:**
- Use Reserved Instances or Savings Plans for ECS Fargate (up to 50% savings)
- Enable RDS auto-scaling for storage
- Use AWS Cost Explorer to track actual costs
- Consider smaller instance types for non-production environments

## Cleanup

To delete all resources:

```bash
aws cloudformation delete-stack --stack-name $STACK_NAME
```

## Troubleshooting

### Tasks not starting

1. Check ECS service events:
```bash
aws ecs describe-services \
  --cluster $STACK_NAME-LiteLLM-Cluster \
  --services $STACK_NAME-litellm-service \
  --query 'services[0].events[0:5]'
```

2. Check task logs:
```bash
aws logs tail /ecs/$STACK_NAME-litellm --follow
```

### Database connection issues

1. Verify RDS is running:
```bash
aws rds describe-db-instances \
  --db-instance-identifier $STACK_NAME-litellm-db \
  --query 'DBInstances[0].DBInstanceStatus'
```

2. Check security group rules allow ECS → RDS communication

### High latency

1. Check if you have enough tasks running:
```bash
aws ecs describe-services \
  --cluster $STACK_NAME-LiteLLM-Cluster \
  --services $STACK_NAME-litellm-service \
  --query 'services[0].[runningCount,desiredCount]'
```

2. Monitor RDS performance in CloudWatch
3. Consider scaling up task count or RDS instance size

## Advanced Configuration

### Adding Redis Cache

Redis can reduce database load by 60-80%. To add Redis:

1. Add ElastiCache Redis cluster to the CloudFormation template
2. Update task environment variables:
```yaml
- Name: REDIS_HOST
  Value: !GetAtt RedisCluster.RedisEndpoint.Address
- Name: REDIS_PORT
  Value: 6379
```
3. Update proxy config to enable caching

### Custom Domain with HTTPS

1. Create an SSL certificate in AWS Certificate Manager
2. Add HTTPS listener to the ALB:
```bash
aws elbv2 create-listener \
  --load-balancer-arn <ALB-ARN> \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=<CERT-ARN> \
  --default-actions Type=forward,TargetGroupArn=<TG-ARN>
```
3. Update Route53 DNS to point to the ALB

### Auto-scaling

Enable ECS Service Auto Scaling based on CPU or request metrics:

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/$STACK_NAME-LiteLLM-Cluster/$STACK_NAME-litellm-service \
  --min-capacity 2 \
  --max-capacity 10

aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/$STACK_NAME-LiteLLM-Cluster/$STACK_NAME-litellm-service \
  --policy-name cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration \
    '{"TargetValue":70.0,"PredefinedMetricSpecification":{"PredefinedMetricType":"ECSServiceAverageCPUUtilization"}}'
```

## Support

- Documentation: https://docs.litellm.ai
- GitHub Issues: https://github.com/BerriAI/litellm/issues
- Benchmark Guide: https://docs.litellm.ai/docs/benchmarks

## References

- [LiteLLM Benchmark Results](https://docs.litellm.ai/docs/benchmarks)
- [AWS ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [AWS RDS Performance Best Practices](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_BestPractices.html)
