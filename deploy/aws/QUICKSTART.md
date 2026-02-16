# Quick Start Guide - AWS Deployment

Deploy LiteLLM on AWS in under 5 minutes with the benchmark configuration.

## Prerequisites

- AWS account with CLI configured
- Bash shell (Linux, macOS, or WSL on Windows)

## 1-Click Deployment

Run the deployment script:

```bash
cd deploy/aws
./deploy.sh
```

The script will:
1. Prompt you for a database password and master key
2. Create all necessary AWS resources (VPC, ECS, RDS, ALB)
3. Deploy 4 LiteLLM instances with 4 workers each
4. Wait for deployment to complete (~10-15 minutes)
5. Display your API endpoint and credentials

## What Gets Deployed

```
┌─────────────────────────────────────────┐
│         Application Load Balancer        │
│              (Public)                    │
└────────────┬────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼───┐         ┌───▼───┐
│ ECS   │         │ ECS   │
│ Task  │  ...    │ Task  │
│ (4    │         │ (4    │
│ vCPU) │         │ vCPU) │
│ 4     │         │ 4     │
│ workers)        │ workers)
└───┬───┘         └───┬───┘
    │                 │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  RDS PostgreSQL │
    │   (db.t3.medium)│
    │      100 GB      │
    └─────────────────┘
```

**Configuration:**
- 4 ECS Fargate tasks (4 vCPU, 8 GB RAM each)
- 4 workers per task = 16 total workers
- PostgreSQL database (db.t3.medium)
- Application Load Balancer
- Private VPC with NAT Gateway

## Using Your Deployment

### Make Your First API Call

```bash
# Set your credentials (from deployment output)
export LITELLM_URL="http://your-alb-url"
export LITELLM_KEY="your-master-key"

# Test the API
curl -X POST "$LITELLM_URL/v1/chat/completions" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fake-openai-endpoint",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Add Real LLM Providers

Update your configuration to use real providers like OpenAI, Anthropic, etc:

```bash
# Get your ECS cluster and service names
CLUSTER=$(aws cloudformation describe-stacks \
  --stack-name litellm-benchmark \
  --query 'Stacks[0].Outputs[?OutputKey==`ECSClusterName`].OutputValue' \
  --output text)

SERVICE=$(aws cloudformation describe-stacks \
  --stack-name litellm-benchmark \
  --query 'Stacks[0].Outputs[?OutputKey==`ECSServiceName`].OutputValue' \
  --output text)

# Update task definition environment variables
# (See README.md for detailed instructions)
```

## Benchmark Your Deployment

Install Locust and run the benchmark test:

```bash
# Install Locust
pip install locust

# Run benchmark (1000 users, 500 spawn rate, 5 minutes)
export LITELLM_MASTER_KEY="your-master-key"
locust -f locustfile.py \
  --host=$LITELLM_URL \
  --users=1000 \
  --spawn-rate=500 \
  --run-time=5m \
  --headless
```

**Expected Results:**
- Median latency: ~100 ms
- P95 latency: ~150 ms
- Throughput: ~1,170 RPS

## Monitoring

View real-time logs:

```bash
aws logs tail /ecs/litellm-benchmark-litellm --follow
```

Monitor key metrics in CloudWatch:
- ECS CPU/Memory utilization
- ALB request count and latency
- RDS connections and CPU

## Cleanup

Delete all resources when done:

```bash
aws cloudformation delete-stack --stack-name litellm-benchmark
```

This will remove all AWS resources and stop charges.

## Cost

**Estimated monthly cost:** ~$440-460

Breakdown:
- ECS Fargate: ~$350
- RDS PostgreSQL: ~$60
- Application Load Balancer: ~$20
- Data Transfer & NAT Gateway: ~$10-30

## Customization

### Scale to 8 instances

```bash
DESIRED_TASKS=8 ./deploy.sh
```

### Use different instance sizes

```bash
TASK_CPU=8192 TASK_MEMORY=16384 ./deploy.sh
```

### Deploy to a different region

```bash
AWS_REGION=us-west-2 ./deploy.sh
```

## Troubleshooting

### Tasks not starting

Check ECS service events:
```bash
aws ecs describe-services \
  --cluster litellm-benchmark-LiteLLM-Cluster \
  --services litellm-benchmark-litellm-service
```

### Health checks failing

The tasks may take 2-3 minutes to become healthy after deployment. Check logs:
```bash
aws logs tail /ecs/litellm-benchmark-litellm --follow
```

### High latency

1. Check if all tasks are running
2. Verify you have the right number of workers
3. Consider scaling up task count or instance size

## Next Steps

- [Full README](README.md) - Complete documentation
- [Benchmark Guide](https://docs.litellm.ai/docs/benchmarks) - Performance details
- [LiteLLM Docs](https://docs.litellm.ai) - Configuration and features

## Support

- GitHub Issues: https://github.com/BerriAI/litellm/issues
- Documentation: https://docs.litellm.ai
